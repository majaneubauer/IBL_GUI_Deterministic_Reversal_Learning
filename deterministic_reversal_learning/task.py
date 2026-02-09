import logging
from pathlib import Path
import yaml
import iblrig
from collections import OrderedDict
from collections.abc import Callable
from enum import IntEnum
from pybpodapi.bpod_modules.bpod_module import BpodModule
from pybpodapi.com.messaging.trial import Trial
from iblutil.io import binary, jsonable
from iblutil.util import Bunch
import numpy as np

from iblrig.base_choice_world import (
    ActiveChoiceWorldSession,
    ActiveChoiceWorldTrialData,
)
from iblrig.misc import get_task_arguments

log = logging.getLogger(__name__)

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath("task_parameters.yaml")) as f:
    DEFAULTS = yaml.safe_load(f)


class DeterministicReversalLearningTrialData(ActiveChoiceWorldTrialData):
    pass


SOFTCODE = IntEnum(
    "SOFTCODE",
    [
        "STOP_SOUND",
        "PLAY_TONE",
        "PLAY_NOISE",
        "PLAY_INSTRUCTIVE_TONE",
        "TRIGGER_CAMERA",
    ],
)


class Session(ActiveChoiceWorldSession):
    protocol_name = "DeterministicReversalLearning"
    TrialDataModel = DeterministicReversalLearningTrialData

    def next_trial(self):
        self.trial_num += 1
        self.draw_next_trial_info(pleft=self.task_params.PROBABILITY_LEFT)

    def get_state_machine_trial(self, i):
        # we define the trial number here for subclasses that may need it
        sma = self._instantiate_state_machine(trial_number=i)

        # Signal trial start and stop all sounds
        sma.add_state(
            state_name="trial_start",
            state_timer=0,  # ~100µs hardware irreducible delay
            state_change_conditions={"Tup": "reset_rotary_encoder"},
            output_actions=[self.bpod.actions.stop_sound, ("BNC1", 255)],
        )

        # Reset the rotary encoder by sending the following opcodes via the modules serial interface
        # - 'Z' (ASCII 90): Set current rotary encoder position to zero
        # - 'E' (ASCII 69): Enable all position thresholds (that may have been disabled by a threshold-crossing)
        # cf. https://sanworks.github.io/Bpod_Wiki/serial-interfaces/rotary-encoder-module-serial-interface/
        sma.add_state(
            state_name="reset_rotary_encoder",
            state_timer=0,
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={"Tup": "quiescent_period"},
        )

        # Quiescent Period. If the wheel is moved past one of the thresholds: Reset the rotary encoder and start over.
        # Continue with the stimulation once the quiescent period has passed without triggering movement thresholds.
        sma.add_state(
            state_name="quiescent_period",
            state_timer=self.quiescent_period,  # TODO why is that here not self.tasks_params?
            output_actions=[],
            state_change_conditions={
                "Tup": "stim_on",
                self.movement_left: "reset_rotary_encoder",
                self.movement_right: "reset_rotary_encoder",
            },
        )

        # Show the visual stimulus. This is achieved by sending a time-stamped byte-message to Bonsai via the Rotary
        # Encoder Module's ongoing USB-stream. Move to the next state once the Frame2TTL has been triggered, i.e.,
        # when the stimulus has been rendered on screen. Use the state-timer as a backup to prevent a stall.
        sma.add_state(
            state_name="stim_on",
            state_timer=0.1,
            output_actions=[
                self.bpod.actions.bonsai_show_stim
            ],  # TODO change to show in centre
            state_change_conditions={
                "BNC1High": "interactive_delay",
                "BNC1Low": "interactive_delay",
                "Tup": "interactive_delay",
            },
        )

        # Defined delay between visual and auditory cue
        sma.add_state(
            state_name="interactive_delay",
            state_timer=self.task_params.INTERACTIVE_DELAY,
            output_actions=[],
            state_change_conditions={"Tup": "play_instructive_tone"},
        )

        # Play instrucive tone. Move to next state if sound is detected. Use the state-timer as a backup to prevent a stall.
        sma.add_state(
            state_name="play_instructive_tone",
            state_timer=0.1,
            output_actions=[
                self.bpod.actions.play_instructive_tone
            ],  # TODO create instructive tone?
            state_change_conditions={
                "Tup": "open_loop",
                "BNC2High": "open_loop",
            },
        )

        # Start the open loop state in which the animal can move the wheel but wheel movement is not coupled to the stimulus position.
        sma.add_state(
            state_name="open_loop",
            state_timer=self.task_params.DECISION_PERIOD_SECS,
            output_actions=[],
            state_change_conditions={"Tup": "play_go_tone"},
        )

        # Play go tone. Move to next state if sound is detected. Use the state-timer as a backup to prevent a stall.
        sma.add_state(
            state_name="play_go_tone",
            state_timer=0.1,
            output_actions=[self.bpod.actions.play_tone],  # create/modify (go)_tone?
            state_change_conditions={
                "Tup": "reset2_rotary_encoder",
                "BNC2High": "reset2_rotary_encoder",
            },
        )

        # Reset rotary encoder (see above). Move on after brief delay (to avoid a race conditions in the bonsai flow).
        sma.add_state(
            state_name="reset2_rotary_encoder",
            state_timer=0.05,
            output_actions=[self.bpod.actions.rotary_encoder_reset],
            state_change_conditions={"Tup": "closed_loop"},
        )

        # Start the closed loop state in which the animal controls the position of the visual stimulus by means of the
        # rotary encoder. The three possible outcomes are:
        # 1) wheel has NOT been moved past a threshold: continue with no-go condition
        # 2) wheel has been moved in WRONG direction: continue with error condition
        # 3) wheel has been moved in CORRECT direction: continue with reward condition

        sma.add_state(
            state_name="closed_loop",
            state_timer=self.task_params.RESPONSE_WINDOW,
            output_actions=[
                self.bpod.actions.bonsai_closed_loop
            ],  # TODO change bonsai closed loop meaning?
            state_change_conditions={
                "Tup": "no_go",
                self.event_error: "freeze_error",
                self.event_reward: "freeze_reward",
            },
        )

        # No-go: hide the visual stimulus and play white noise. Go to exit_state after FEEDBACK_NOGO_DELAY_SECS.
        sma.add_state(
            state_name="no_go",
            state_timer=self.feedback_nogo_delay,  # TODO why is that here not self.tasks_params?
            output_actions=[
                self.bpod.actions.bonsai_hide_stim,
                self.bpod.actions.play_noise,
            ],
            state_change_conditions={"Tup": "exit_state"},
        )

        # Error: Freeze the stimulus and play white noise.
        # Continue to hide_stim/exit_state once FEEDBACK_ERROR_DELAY_SECS have passed.
        sma.add_state(
            state_name="freeze_error",
            state_timer=0,
            output_actions=[self.bpod.actions.bonsai_freeze_stim],
            state_change_conditions={"Tup": "error"},
        )
        sma.add_state(
            state_name="error",
            state_timer=self.feedback_error_delay,
            output_actions=[self.bpod.actions.play_noise],
            state_change_conditions={"Tup": "hide_stim"},
        )

        # Reward: open the valve for a defined duration (and set BNC1 to high), freeze stimulus (freeze_reward same as
        # freeze_error but #TODO how to differentiate which state to go to when Tup?).
        # Continue to hide_stim/exit_state once FEEDBACK_CORRECT_DELAY_SECS have passed.
        sma.add_state(
            state_name="freeze_reward",
            state_timer=0,
            output_actions=[self.bpod.actions.bonsai_freeze_stim],
            state_change_conditions={"Tup": "reward"},
        )
        sma.add_state(
            state_name="reward",
            state_timer=self.reward_time,
            output_actions=[("Valve1", 255), ("BNC1", 255)],
            state_change_conditions={"Tup": "correct"},
        )
        sma.add_state(
            state_name="correct",
            state_timer=self.feedback_correct_delay - self.reward_time,
            output_actions=[],
            state_change_conditions={"Tup": "hide_stim"},
        )

        # Hide the visual stimulus. This is achieved by sending a time-stamped byte-message to Bonsai via the Rotary
        # Encoder Module's ongoing USB-stream. Move to the next state once the Frame2TTL has been triggered, i.e.,
        # when the stimulus has been rendered on screen. Use the state-timer as a backup to prevent a stall.
        sma.add_state(
            state_name="hide_stim",
            state_timer=0.1,
            output_actions=[self.bpod.actions.bonsai_hide_stim],
            state_change_conditions={
                "Tup": "exit_state",
                "BNC1High": "exit_state",
                "BNC1Low": "exit_state",
            },
        )

        # Wait for ITI_DELAY_SECS before ending the trial. Raise BNC1 to mark this event.
        sma.add_state(
            state_name="exit_state",
            state_timer=self.task_params.ITI_DELAY_SECS,
            output_actions=[("BNC1", 255)],
            state_change_conditions={"Tup": "exit"},
        )

        return sma

    def define_harp_sounds_actions(
        self,
        module: BpodModule,
        go_tone_index: int = 2,
        noise_index: int = 3,
        instructive_tone_index: int = 4,
    ) -> None:
        module_port = f'Serial{module.serial_port if module is not None else ""}'

        self.bpod.actions.update(
            {
                "play_tone": (
                    module_port,
                    self._define_message(module, [ord("P"), go_tone_index]),
                ),
                "play_noise": (
                    module_port,
                    self._define_message(module, [ord("P"), noise_index]),
                ),
                "play_instructive_tone": (
                    module_port,
                    self._define_message(module, [ord("P"), instructive_tone_index]),
                ),
                "stop_sound": (module_port, ord("X")),
            }
        )


    def softcode_dictionary(self) -> OrderedDict[int, Callable]:
        """
        Returns a softcode handler dict where each key corresponds to the softcode and each value to the
        function to be called.

        This needs to be wrapped this way because
            1) we want to be able to inherit this and dynamically add softcode to the dictionry
            2) we need to provide the Task object (self) at run time to have the functions with static args
        This is tricky as it is unclear if the task object is a copy or a reference when passed here.


        Returns
        -------
        OrderedDict[int, Callable]
            Softcode dictionary
        """
        softcode_dict = OrderedDict(
            {
                SOFTCODE.STOP_SOUND: self.sound["sd"].stop,
                SOFTCODE.PLAY_TONE: lambda: self.sound["sd"].play(
                    self.sound["GO_TONE"], self.sound["samplerate"]
                ),
                SOFTCODE.PLAY_INSTRUCTIVE_TONE: lambda: self.sound["sd"].play(
                    self.sound["INSTRUCTIVE_TONE"], self.sound["samplerate"]
                ),
                SOFTCODE.PLAY_NOISE: lambda: self.sound["sd"].play(
                    self.sound["WHITE_NOISE"], self.sound["samplerate"]
                ),
                SOFTCODE.TRIGGER_CAMERA: getattr(
                    self,
                    "trigger_bonsai_cameras",
                    lambda: self._raise_on_undefined_softcode_handler(
                        SOFTCODE.TRIGGER_CAMERA
                    ),
                ),
            }
        )
        return softcode_dict

    def init_mixin_sound(self, *args, **kwargs):
        # 1) let the base class create GO_TONE, WHITE_NOISE, device, etc.
        super().init_mixin_sound(*args, **kwargs)

        # 2) define your new sound
        self.sound["INSTRUCTIVE_TONE"] = iblrig.sound.make_sound(
            rate=self.sound["samplerate"],
            frequency=self.task_params.INSTRUCTIVE_TONE_FREQUENCY,
            duration=self.task_params.INSTRUCTIVE_TONE_DURATION,
            amplitude=self.task_params.INSTRUCTIVE_TONE_AMPLITUDE,
            fade=0.01,
            chans=self.sound["channels"],
        )

    def sound_play_instructive_tone(self, state_timer=0.102, state_name='play_instructive_tone'):
        """
        Play the ready tone beep using bpod state machine.
        :return: bpod current trial export
        """
        return self._sound_play(state_name=state_name, output_actions=[self.bpod.actions.play_tone], state_timer=state_timer)


if __name__ == "__main__":  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

import logging
from pathlib import Path
import yaml
import iblrig
from iblrig.hifi import HiFi
import numpy as np
from iblrig.hardware import RotaryEncoderModule
from pybpodapi.protocol import StateMachine
from pydantic import NonNegativeFloat

from iblrig.base_choice_world import (
    ChoiceWorldSession,
    ChoiceWorldTrialData,
)
from iblrig.misc import get_task_arguments

from iblrig.base_tasks import OSCClient

log = logging.getLogger(__name__)

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath("task_parameters.yaml")) as f:
    DEFAULTS = yaml.safe_load(f)


class HabituationDeterministicReversalLearningTrialData(ChoiceWorldTrialData):
    """Pydantic Model for Trial Data, extended from :class:`~.iblrig.base_choice_world.ChoiceWorldTrialData`."""
    delay_to_stim_end_position: NonNegativeFloat
    stim_end_position: int

class ExtendedOSCClient(OSCClient):
    OSC_PROTOCOL = {
        **OSCClient.OSC_PROTOCOL,
        'stim_end_position': dict(mess='/m', type=int),
    }

class Session(ChoiceWorldSession):
    protocol_name = (
        "HabituationDeterministicReversalLearning"  # here defined how it shows up in GUI
    )
    TrialDataModel = HabituationDeterministicReversalLearningTrialData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # to help bonsai find Gabor2D_MN.bonsai file
        self.paths["VISUAL_STIM_FOLDER"] = self.get_task_directory().parent

    def init_mixin_sound(self):
        # call the original method so that GO_TONE and WHITE_NOISE are initialised as before
        super().init_mixin_sound()
        # determine the amp_gain_factor like it is done in the original method
        if (
            self.hardware_settings.device_sound.OUTPUT == "hifi"
            and self.hardware_settings.device_sound.AMP_TYPE == "AMP2X15"
        ):
            amp_gain_factor = 0.25
        else:
            amp_gain_factor = 1.0
        self.task_params.INSTRUCTIVE_TONE_AMPLITUDE *= (
            amp_gain_factor  # TODO is this a bug, or intentional?
        )
        # create instructive sound
        self.sound["INSTRUCTIVE_TONE"] = iblrig.sound.make_sound(
            rate=self.sound["samplerate"],
            frequency=self.task_params.INSTRUCTIVE_TONE_FREQUENCY,
            duration=self.task_params.INSTRUCTIVE_TONE_DURATION,
            amplitude=self.task_params.INSTRUCTIVE_TONE_AMPLITUDE
            * amp_gain_factor,  # TODO is this a bug, or intentional?
            fade=0.01,
            chans=self.sound["channels"],
        )

    def start_mixin_sound(self):
        super().start_mixin_sound()
        output_type = self.hardware_settings.device_sound["OUTPUT"]
        match output_type:
            case "hifi":
                module = self.bpod.get_module("^HiFi")
                hifi = HiFi(
                    port=self.hardware_settings.device_sound.COM_SOUND,
                    sampling_rate_hz=self.sound["samplerate"],
                )
                # Load the three buffers
                hifi.load(
                    index=self.task_params.GO_TONE_IDX, data=self.sound["GO_TONE"]
                )
                hifi.load(
                    index=self.task_params.INSTRUCTIVE_TONE_IDX,
                    data=self.sound["INSTRUCTIVE_TONE"],
                )
                hifi.load(
                    index=self.task_params.WHITE_NOISE_IDX,
                    data=self.sound["WHITE_NOISE"],
                )
                hifi.push()
                hifi.close()
                # standard two actions
                self.bpod.define_harp_sounds_actions(
                    module=module,
                    go_tone_index=self.task_params.GO_TONE_IDX,
                    noise_index=self.task_params.WHITE_NOISE_IDX,
                )
                # add instructive tone manually
                module_port = f"Serial{module.serial_port}"
                self.bpod.actions.update(
                    {
                        "play_instructive_tone": (
                            module_port,
                            self.bpod._define_message(
                                module,
                                [ord("P"), self.task_params.INSTRUCTIVE_TONE_IDX],
                            ),
                        ),
                    }
                )
        log.info(
            f"Sound module loaded: OK: {self.hardware_settings.device_sound['OUTPUT']}"
        )

    def init_mixin_rotary_encoder(self):
        thresholds_deg = (
            self.task_params.STIM_END_POSITIONS + self.task_params.QUIESCENCE_THRESHOLDS
        )  # STIM_POSITIONS does not need to be in here
        self.device_rotary_encoder = RotaryEncoderModule(
            self.hardware_settings.device_rotary_encoder,
            thresholds_deg,
            self.stimulus_gain,
        )

    def start_mixin_bpod(self):
        super().start_mixin_bpod()
        module = self.bpod.rotary_encoder
        module_port = f'Serial{module.serial_port}'
        self.bpod.actions.update(
            {
                'bonsai_show_end_position': (module_port, self.bpod._define_message(module, [ord('#'), 10])),
            }
        )

    def next_trial(self):
        self.trial_num += 1
        self.draw_next_trial_info()

    def init_mixin_bonsai_visual_stimulus(self, *args, **kwargs):
        # camera 7111, microphone 7112
        self.bonsai_visual_udp_client = ExtendedOSCClient(port=7110)

    def draw_next_trial_info(self, *args, **kwargs):
        # update trial table fields specific to habituation choice world
        self.trials_table.at[self.trial_num, 'delay_to_stim_end_position'] = np.random.normal(self.task_params.DELAY_TO_STIM_END_POSITION, 2)
        # select stim end position
        self.trials_table.at[self.trial_num, 'stim_end_position'] = int(
            np.random.choice(
                self.task_params.STIM_END_POSITIONS,
                p=[
                    self.task_params.PROBABILITY_LEFT,
                    1 - self.task_params.PROBABILITY_LEFT,
                ],
            )
        )
        super().draw_next_trial_info()

    def get_state_machine_trial(self, i):
        sma = StateMachine(self.bpod)

        # NB: This state actually the inter-trial interval, i.e. the period of grey screen between stim off and stim on.
        # During this period the Bpod TTL is HIGH and there are no stimuli. The onset of this state is trial end;
        # the offset of this state is trial start!
        sma.add_state(
            state_name='iti',
            state_timer=1,  # Stim off for 1 sec
            state_change_conditions={'Tup': 'stim_on'},
            output_actions=[self.bpod.actions.bonsai_hide_stim, ('BNC1', 255)],
        )

        # This stim_on state is considered the actual trial start
        sma.add_state(
            state_name="stim_on",
            state_timer=0.1,
            output_actions=[self.bpod.actions.bonsai_show_stim],
            state_change_conditions={
                "BNC1High": "play_instructive_tone",
                "BNC1Low": "play_instructive_tone",
                "Tup": "play_instructive_tone",
            },
        )

        sma.add_state(
            state_name="play_instructive_tone",
            state_timer=0.1,
            output_actions=[self.bpod.actions.play_instructive_tone],
            state_change_conditions={
                "Tup": "open_loop",
                "BNC2High": "open_loop",
            },
        )

        sma.add_state(
            state_name="open_loop",
            state_timer=self.task_params.DECISION_PERIOD_SECS,
            output_actions=[],
            state_change_conditions={"Tup": "play_go_tone"},
        )

        sma.add_state(
            state_name="play_go_tone",
            state_timer=0.1,
            output_actions=[self.bpod.actions.play_tone],
            state_change_conditions={
                "Tup": "stim_on_delay",
                "BNC2High": "stim_on_delay",
            },
        )

        sma.add_state(
            state_name='stim_on_delay',
            state_timer=self.trials_table.at[self.trial_num, 'delay_to_stim_end_position'],
            state_change_conditions={'Tup': 'stim_end_position'},
            output_actions=[],
        )

        sma.add_state(
            state_name='stim_end_position',
            state_timer=0.5,
            state_change_conditions={'Tup': 'reward'},
            output_actions=[self.bpod.actions.bonsai_show_end_position],
        )

        sma.add_state(
            state_name='reward',
            state_timer=self.reward_time,  # the length of time to leave reward valve open, i.e. reward size
            state_change_conditions={'Tup': 'post_reward'},
            output_actions=[('Valve1', 255), ('BNC1', 255)],
        )
        # This state defines the period after reward where Bpod TTL is LOW.
        # NB: The stimulus is on throughout this period. The stim off trigger occurs upon exit.
        # The stimulus thus remains in the screen centre for 0.5 + ITI_DELAY_SECS seconds.
        sma.add_state(
            state_name='post_reward',
            state_timer=self.task_params.ITI_DELAY_SECS - self.reward_time,
            state_change_conditions={'Tup': 'exit'},
            output_actions=[],
        )
        return sma


if __name__ == "__main__":  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

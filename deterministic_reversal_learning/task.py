import logging
from pathlib import Path
import yaml
import iblrig
from iblrig.hifi import HiFi
import numpy as np
from typing import Any
from re import split as re_split
import matplotlib.pyplot as plt
from scipy.stats import beta
import pyqtgraph as pg
from qtpy.QtWidgets import QApplication
from iblrig.hardware import RotaryEncoderModule

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
    """Pydantic Model for Trial Data, extended from :class:`~.iblrig.base_choice_world.ActiveChoiceWorldTrialData`."""

    block_side: int  # -1 for left or +1 for right
    alpha: float
    beta: float
    map_probability: float
    precision: float
    success_total: float
    failure_total: float


class Session(ActiveChoiceWorldSession):
    protocol_name = (
        "DeterministicReversalLearning"  # here defined how it shows up in GUI
    )
    TrialDataModel = DeterministicReversalLearningTrialData

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # to help bonsai find Gabor2D_MN.bonsai file
        self.paths["VISUAL_STIM_FOLDER"] = self.get_task_directory()
        # add block state
        self.block_side = int(
            np.random.choice(
                self.task_params.BLOCK_SIDES,
                p=[
                    self.task_params.PROBABILITY_LEFT,
                    1 - self.task_params.PROBABILITY_LEFT,
                ],
            )
        )  # -1 = left, +1 = right
        self.block_length = self.task_params.BLOCK_LENGTH
        self.block_trial_counter = (
            -1
        )  # needs to be -1 and not 0 for next_trial condition to work

        # initialise online plots
        self.init_online_plot()

        # initialise bsa online plots
        self.bsa_plot = None
        self.bsa_curve = None

    # initialise online plots
    def init_online_plot(self):
        plt.ion()  # interactive mode ON
        self.fig, self.ax = plt.subplots(figsize=(10, 5))
        (self.line_map,) = self.ax.plot([], [], lw=0.75, label="MAP probability")
        self.ax.axhline(
            y=0.5, color="firebrick", linestyle="--", linewidth=0.75, label="Chance"
        )
        self.ax.set_xlim(0, self.task_params.NTRIALS)
        self.ax.set_ylim(0, 1.25)
        self.ax.set_xlabel("Trial")
        self.ax.set_ylabel("P(Strategy)")
        self.ax.legend()
        plt.show()

    def attach_bsa_plot(self):
        app = QApplication.instance()
        for widget in app.topLevelWidgets():
            if widget.windowTitle() == "Online Plots":
                self.online_window = widget
                break
        else:
            self.online_window = None

        # modify layout of self.online_window
        if self.online_window is not None:
            layout = self.online_window.centralWidget().layout()
            # remove psychometric and chronometric plots
            self.online_window.psychometricWidget.setParent(None)
            self.online_window.chronometricWidget.setParent(None)
            # create own plot
            self.bsa_plot = pg.PlotWidget()
            self.bsa_plot.setBackground("w")
            self.bsa_plot.setTitle("Bayesian Strategy Analysis - Correct Choice")
            self.bsa_plot.setLabel("left", "P(Strategy)")
            self.bsa_plot.setLabel("bottom", "Trial")
            self.bsa_plot.setYRange(0, 1.25)
            self.bsa_curve = self.bsa_plot.plot(pen=pg.mkPen(width=2))

            # add into layout at same position
            layout.addWidget(self.bsa_plot, 1, 1, 2, 1)

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

    @property
    def correct_end_position(self):
        return int(
            self.block_side * abs(self.task_params.STIM_END_POSITIONS[0])
        )  # left block: -1 * |-35| = -35; right block: 1 * |-35| = 35

    @property
    def event_error(self):
        return self.device_rotary_encoder.THRESHOLD_EVENTS[
            (1 if self.task_params.STIM_REVERSE else -1) * self.correct_end_position
        ]

    @property
    def event_reward(self):
        return self.device_rotary_encoder.THRESHOLD_EVENTS[
            (-1 if self.task_params.STIM_REVERSE else 1) * self.correct_end_position
        ]

    def next_trial(self):
        self.trial_num += 1

        # update block counter
        self.block_trial_counter += 1

        # deterministic reversal
        if (
            self.block_trial_counter >= self.block_length
        ):  # TODO change reversal criterion here
            self.block_side *= -1  # flip block: -1*-1 = 1; 1*-1 = -1
            self.block_trial_counter = 0
            log.warning(
                f"Reversal! New block side: {self.block_side}"
            )  # does not work with log.info

        self.draw_next_trial_info(
            pleft=self.task_params.PROBABILITY_LEFT
        )  # no need to change anything here, because stimulus position is [0, 0]

        # log block side
        self.trials_table.at[self.trial_num, "block_side"] = self.block_side

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
            output_actions=[self.bpod.actions.bonsai_show_stim],
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
            output_actions=[self.bpod.actions.play_instructive_tone],
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
            output_actions=[self.bpod.actions.play_tone],
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

    def trial_completed(self, bpod_data: dict) -> None:
        # removed assertion error for position = 0 cause that is what we want
        # Get the response time from the behaviour data.
        # It is defined as the time passing between the start of `stim_on` and the end of `closed_loop`.
        state_times = bpod_data["States timestamps"]
        response_time = state_times["closed_loop"][0][1] - state_times["stim_on"][0][0]
        self.trials_table.at[self.trial_num, "response_time"] = response_time

        try:
            # Get the trial's outcome, i.e., the states that have a matching name and a valid time-stamp
            # Assert that we have exactly one outcome
            outcome_names = [
                "correct",
                "error",
                "no_go",
                "omit_correct",
                "omit_error",
                "omit_no_go",
            ]
            outcomes = [
                name
                for name, times in state_times.items()
                if name in outcome_names and ~np.isnan(times[0][0])
            ]
            if (n_outcomes := len(outcomes)) != 1:
                trial_states = "Trial states: " + ", ".join(
                    k for k, v in state_times.items() if ~np.isnan(v[0][0])
                )
                assert (
                    n_outcomes != 0
                ), f"No outcome detected for trial {self.trial_num}.\n{trial_states}"
                assert (
                    n_outcomes == 1
                ), f"{n_outcomes} outcomes detected for trial {self.trial_num}.\n{trial_states}"
            outcome = outcomes[0]

        except AssertionError as e:
            # write bpod_data to disk, log exception then raise
            self.save_trial_data_to_json(bpod_data, validate=False)
            for line in re_split(r"\n", e.args[0]):
                log.error(line)
            raise e

        # record the trial's outcome in the trials_table
        self.trials_table.at[self.trial_num, "trial_correct"] = "correct" in outcome
        if "correct" in outcome:
            self.session_info.NTRIALS_CORRECT += 1
            self.trials_table.at[self.trial_num, "response_side"] = np.sign(
                self.correct_end_position
            )  # np.sign returns 1 for positive and -1 for negative numbers
        elif "error" in outcome:
            self.trials_table.at[self.trial_num, "response_side"] = -np.sign(
                self.correct_end_position
            )
        elif "no_go" in outcome:
            self.trials_table.at[self.trial_num, "response_side"] = 0

        # attach plot once
        if not hasattr(self, "bsa_plot"):
            self.attach_bsa_plot()

        # run bayesian strategy analysis
        self.bayesian_strategy_analysis()

        super(ActiveChoiceWorldSession, self).trial_completed(bpod_data)

    def show_trial_log(
        self, extra_info: dict[str, Any] | None = None, log_level: int = logging.INFO
    ):
        # construct info dict
        trial_info = self.trials_table.iloc[self.trial_num]
        info_dict = {
            "Block Side": f"{trial_info.block_side}",
            "Correct End Position": self.correct_end_position,
            "N Trials Block": self.block_trial_counter
            + 1,  # +1 because counter starts at 0
            "Alpha": trial_info.alpha,
            "Beta": trial_info.beta,
            "MAP Probability": trial_info.map_probability,
            "Precision": trial_info.precision,
            "Success Total": trial_info.success_total,
            "Failure Total": trial_info.failure_total,
        }

        # update info dict with extra_info dict
        if isinstance(extra_info, dict):
            info_dict.update(extra_info)

        # call parent method
        super().show_trial_log(extra_info=info_dict, log_level=log_level)

    def _finalize(self):
        # first let parent clean up (kills plotting subprocess etc.)
        super()._finalize()

        # save only completed trials
        df = self.trials_table.iloc[: self.trial_num + 1]

        output_path = self.paths.SESSION_FOLDER / "trials_table.csv"
        df.to_csv(output_path, index=False)

        log.info(f"Trials table saved to {output_path}")

    @staticmethod
    def set_priors(prior_type: str):
        if prior_type == "Uniform":
            alpha0, beta0 = 1, 1
        elif prior_type == "Jeffreys":
            alpha0, beta0 = 0.5, 0.5
        else:
            raise ValueError(f"Unknown prior: {prior_type}")
        return alpha0, beta0

    @staticmethod
    def update_strategy_posterior_probability(
        trial_type, decay_rate, success_total, failure_total, alpha0, beta0
    ):
        if trial_type == "success":
            success_total = decay_rate * success_total + 1
            failure_total = decay_rate * failure_total
            alpha = alpha0 + success_total
            beta = beta0 + failure_total
        elif trial_type == "failure":
            success_total = decay_rate * success_total
            failure_total = decay_rate * failure_total + 1
            alpha = alpha0 + success_total
            beta = beta0 + failure_total
        else:
            alpha = np.nan
            beta = np.nan

        return success_total, failure_total, alpha, beta

    @staticmethod
    def summaries_of_beta_distribution(alpha, beta_params, stat_type, *args):
        if np.isnan(alpha) or np.isnan(beta_params):
            statistic = np.nan
        else:
            if stat_type == "MAP":
                x = np.arange(0, 1, 0.001)
                y = beta.pdf(x, alpha, beta_params)
                statistic = x[np.argmax(y)]
            elif stat_type == "Mean":
                statistic = alpha / (alpha + beta_params)
            elif stat_type == "Var":
                statistic = (alpha * beta_params) / (
                    ((alpha + beta_params) ** 2) * (alpha + beta_params + 1)
                )
            elif stat_type == "precision":
                statistic = 1 / (
                    (alpha * beta_params)
                    / (((alpha + beta_params) ** 2) * (alpha + beta_params + 1))
                )
            elif stat_type == "Percentile":
                Prct = args[0]
                Delta = 1 - Prct
                P = [Delta / 2, Prct + Delta / 2]
                statistic = beta.ppf(P, alpha, beta_params)
            else:
                print("ERROR")
        return statistic

    def bayesian_strategy_analysis(self):
        if self.trial_num == 0:
            self.prior_type = "Uniform"  # choose prior type
            self.alpha0, self.beta0 = self.set_priors(self.prior_type)  # define priors
            self.decay_rate = 0.9  # set decay rate (gamma)
            # initialise variables to zero
            self.success_total = 0
            self.failure_total = 0
            # initialise list to store MAP probabilities
            self.map_data = []

        # get current trial only
        row = self.trials_table.iloc[self.trial_num]
        trial_type = "success" if row["trial_correct"] else "failure"

        # update once
        self.success_total, self.failure_total, alpha, beta = (
            self.update_strategy_posterior_probability(
                trial_type,
                self.decay_rate,
                self.success_total,
                self.failure_total,
                self.alpha0,
                self.beta0,
            )
        )

        map_probability = self.summaries_of_beta_distribution(alpha, beta, "MAP")
        precision = self.summaries_of_beta_distribution(alpha, beta, "precision")

        # store only current trial
        self.trials_table.at[self.trial_num, "alpha"] = alpha
        self.trials_table.at[self.trial_num, "beta"] = beta
        self.trials_table.at[self.trial_num, "map_probability"] = map_probability
        self.trials_table.at[self.trial_num, "precision"] = precision
        self.trials_table.at[self.trial_num, "success_total"] = self.success_total
        self.trials_table.at[self.trial_num, "failure_total"] = self.failure_total

        # online plots
        self.map_data.append(map_probability)  # add latest MAP
        self.line_map.set_data(range(len(self.map_data)), self.map_data)  # update line
        self.ax.relim()  # recompute axis limits
        self.ax.autoscale_view()  # rescale axes
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

        # online plots widget
        if hasattr(self, "bsa_curve"):
            trials = np.arange(self.trial_num + 1)
            map_probs = self.trials_table["map_probability"].iloc[: self.trial_num + 1]
            self.bsa_curve.setData(trials, map_probs)


if __name__ == "__main__":  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

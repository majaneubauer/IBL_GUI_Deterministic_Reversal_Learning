import logging
from pathlib import Path
import yaml
from iblrig.misc import get_task_arguments


from drl_tasks.deterministic_reversal_learning import (
    HabituationDeterministicReversalLearningSession,
)

log = logging.getLogger(__name__)

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath("task_parameters.yaml")) as f:
    DEFAULTS = yaml.safe_load(f)


class Session(HabituationDeterministicReversalLearningSession):
    protocol_name = (
        "HabituationDeterministicReversalLearning"  # here defined how it shows up in GUI
    )

    def __init__(
        self,
        *args,
        reward_amount_ul: float = DEFAULTS['REWARD_AMOUNT_UL'],
        stop_miniscope_secs: int = DEFAULTS['STOP_MINISCOPE_SECS'],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.task_params['REWARD_AMOUNT_UL'] = reward_amount_ul
        self.task_params['STOP_MINISCOPE_SECS'] = stop_miniscope_secs

    @staticmethod
    def extra_parser():
        """:return: argparse.parser()"""
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--reward_amount_ul',
            option_strings=['--reward_amount_ul'],
            dest='reward_amount_ul',
            default=DEFAULTS['REWARD_AMOUNT_UL'],
            type=float,
            help='Reward amount (µl) within one session',
        )
        parser.add_argument(
            '--stop_miniscope_secs',
            option_strings=['--stop_miniscope_secs'],
            dest='stop_miniscope_secs',
            default=DEFAULTS['STOP_MINISCOPE_SECS'],
            type=int,
            help='Length of HIGH signal to stop miniscope at the end of one session',
        )
        return parser


if __name__ == "__main__":  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

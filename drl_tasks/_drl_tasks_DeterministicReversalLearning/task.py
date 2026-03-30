import logging
from pathlib import Path
import yaml
from iblrig.misc import get_task_arguments

from drl_tasks.deterministic_reversal_learning import (
    DeterministicReversalLearningSession,
)

log = logging.getLogger(__name__)

# read defaults from task_parameters.yaml
with open(Path(__file__).parent.joinpath("task_parameters.yaml")) as f:
    DEFAULTS = yaml.safe_load(f)


class Session(DeterministicReversalLearningSession):
    protocol_name = (
        "DeterministicReversalLearning"  # here defined how it shows up in GUI
    )


    @staticmethod
    def extra_parser():
        """:return: argparse.parser()"""
        parser = super(Session, Session).extra_parser()
        parser.add_argument(
            '--block_length',
            option_strings=['--block_length'],
            dest='block_length',
            default=DEFAULTS['BLOCK_LENGTH'],
            type=int,
            help='Length of a block within one session',
        )
        parser.add_argument(
            '--contrast_set',
            option_strings=['--contrast_set'],
            dest='contrast_set',
            default=DEFAULTS['CONTRAST_SET'],
            nargs='+',
            type=float,
            help='Set of contrasts to present',
        )
        parser.add_argument(
            '--stim_gain',
            option_strings=['--stim_gain'],
            dest='stim_gain',
            default=DEFAULTS['STIM_GAIN'],
            type=float,
            help=f'Visual angle/wheel displacement (deg/mm, default: {DEFAULTS["STIM_GAIN"]})',
        )
        return parser


if __name__ == "__main__":  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

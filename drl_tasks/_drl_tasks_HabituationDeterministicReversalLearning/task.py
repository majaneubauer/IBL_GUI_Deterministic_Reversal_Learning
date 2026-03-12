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


if __name__ == "__main__":  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

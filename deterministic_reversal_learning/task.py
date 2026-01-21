import logging
from pathlib import Path

import yaml

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


class Session(ActiveChoiceWorldSession):
    protocol_name = "DeterministicReversalLearning"
    TrialDataModel = DeterministicReversalLearningTrialData
    # TODO: Implementation


if __name__ == "__main__":  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

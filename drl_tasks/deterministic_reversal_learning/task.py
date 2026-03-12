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
from qtpy.QtCore import QTimer
import pyqtgraph as pg
from qtpy.QtWidgets import QWidget, QVBoxLayout
from iblrig.hardware import RotaryEncoderModule
from iblrig.gui.online_plots import OnlinePlotsView

from iblrig.base_choice_world import (
    ActiveChoiceWorldSession,
    ActiveChoiceWorldTrialData,
)
from iblrig.misc import get_task_arguments

from drl_tasks.DRL import (
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


if __name__ == "__main__":  # pragma: no cover
    kwargs = get_task_arguments(parents=[Session.extra_parser()])
    sess = Session(**kwargs)
    sess.run()

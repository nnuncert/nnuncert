from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import enum
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap

import nnuncert.app.uci.data
from nnuncert.app.uci.data import UCI_DATASETS, UCI, BostonHousing, \
    ConcreteStrength, EnergyEfficiency, Kin8nm, NavalPropulsion, \
    PowerPlant, ProteinStucture, WineQualityRed, YachtHydrodynamics
import nnuncert.app.uci.tracking
from nnuncert.app.uci.tracking import UCIRun, UCIResults
from nnuncert.app.uci.plotting import uci_boxplot
from nnuncert.utils.traintest import generate_random_split, make_gap_splits, \
    make_tail_splits


MODEL_COLORS = {
    "MC Dropout" : get_cmap("tab20").colors[0],       # blue
    "MC dropout" : get_cmap("tab20").colors[0],       # blue
    "MC dropout 200" : get_cmap("tab20").colors[0],   # blue
    "MC dropout 400" : get_cmap("tab20").colors[0],   # blue
    'PBP': "#52b2bf",                                 # turquoise
    'PNN' : "#d8b709",                                # yellow
    'PNN-E' : "#d87409",                              # orange
    'NLM' : get_cmap("tab20").colors[5],              # light green
    'NLM-E' : get_cmap("tab20").colors[4],            # medium green
    'DNNC-R' : "#972d14",                             # winered
    'DNNC-R-STDNN' : get_cmap("tab20").colors[0],     # blue
}


def show_model_colors():
    """Plot all model colors as barplot.

    """
    fig, ax = plt.subplots()
    for i , (m, c) in enumerate(MODEL_COLORS.items()):
        ax.bar(m, 1, color=c)
    ax.tick_params(axis='x', rotation=90)
    ax.set_yticklabels([])


def load_uci(name: str) -> UCI:
    """Load a UCI dataset.

    Parameters
    ----------
    name : str
        Name of data to load.

    Returns
    -------
    UCI

    """
    uci = {
        "boston": BostonHousing(),
        "concrete": ConcreteStrength(),
        "energy": EnergyEfficiency(),
        "kin8nm": Kin8nm(),
        "naval": NavalPropulsion(),
        "powerplant": PowerPlant(),
        "protein": ProteinStucture(),
        "wine": WineQualityRed(),
        "yacht": YachtHydrodynamics(),
    }[name]

    return uci


def load_all() -> List[UCI]:
    """Load all UCI datasets.

    Returns
    -------
    List[UCI]

    """
    return [load_uci(d) for d in UCI_DATASETS]


class Mode(enum.IntEnum):
    LOAD_SCORES = 0
    LOAD_MODELS = 1
    LOAD_MODELS_AND_SAVE = 2
    CALC = 3
    SAVE_SCORES = 4
    SAVE_SCORES_AND_MODEL = 5

    @classmethod
    def _from_str(self, mode_str: str):
        return {
            "load_scores" : Mode.LOAD_SCORES,
            "load_model" : Mode.LOAD_MODELS,
            "load_model_and_save" : Mode.LOAD_MODELS_AND_SAVE,
            "calc" : Mode.CALC,
            "save_scores" : Mode.SAVE_SCORES,
            "save_scores_and_model" : Mode.SAVE_SCORES_AND_MODEL,
        }[mode_str]


class RunType(enum.IntEnum):
    RANDOM = 0
    GAP = 1
    TAIL = 2
    CALIBRATION = 3

    @classmethod
    def _from_str(self, s_str: str):
        return {
            "random" : RunType.RANDOM,
            "gap" : RunType.GAP,
            "tail" : RunType.TAIL,
            "calibration" : RunType.CALIBRATION,
        }[s_str]

    def get_dir_name(self):
        return {
            RunType.RANDOM : "random",
            RunType.GAP : "gap",
            RunType.TAIL : "tail",
            RunType.CALIBRATION : "calibration",
        }[self]

    def make_splits(self,
                    uci: UCI,
                    max_splits: int = 20,
                    ratio: float = 0.1,
                    rng: Union[None, int, np.random.Generator] = None,
                    *args, **kwargs) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Generate splits for UCI dataset.

        Parameters
        ----------
        uci : UCI
            Dataset that generates splits.
        max_splits : int
            Number of splits, only valid for random splits.
        ratio : float
            Test size, default = 0.1.
        rng : Union[None, int, np.random.Generator]
            Random number generator, default = None.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray]]
            List of tuples(train_set, test_set).

        """
        rng = np.random.default_rng(rng)
        if self == RunType.RANDOM:
            s = [generate_random_split(uci.data.y, ratio=ratio, rng=rng,
                                       *args, **kwargs)
                 for _ in range(max_splits)]
        elif self == RunType.GAP:
            s = uci.make_gap_splits(*args, **kwargs)
        elif self == RunType.TAIL:
            s = uci.make_tail_splits(*args, **kwargs)
        elif self == RunType.CALIBRATION:
            s = [generate_random_split(uci.data.y, ratio=0, rng=rng, *args,
                                       **kwargs)]
        return s


def get_uci_path(dir_: str,
                 dataset: str,
                 model: str,
                 i: Union[int, str],
                 run_type: RunType) -> str:
    """Get path of directory where model/scores are (to be) saved.

    Parameters
    ----------
    dir_ : str
        Base directory.
    dataset : str
        Dataset name.
    model : str
        Model name.
    i : Union[int, str]
        Iteration number.
    run_type : RunType
        Random / Gap / Tail / Calibration run.

    Returns
    -------
    str
        Directory where model/scores are (to be) saved.

    """
    if isinstance(i, int):
        i = str(i)
    if run_type == RunType.CALIBRATION:
        return os.path.join(*[dir_, run_type.get_dir_name(), dataset, model])
    return os.path.join(*[dir_, run_type.get_dir_name(), dataset, model, i])

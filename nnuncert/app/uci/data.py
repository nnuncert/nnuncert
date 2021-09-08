from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional
import os
import copy

import pandas as pd
import numpy as np

from nnuncert.utils.traintest import TrainTestSplit, make_gap_splits, make_tail_splits


UCI_DATASETS = ["boston", "concrete", "energy", "kin8nm", "naval","powerplant",
                "protein", "wine", "yacht"]


class UciData():
    """Class to handle the data for a UCI dataset.

    Parameters
    ----------
    csv_dir : str
        Path to load .csv data from.
    response_var : str
        Name of response variable.
    categoricals : List[str]
        List of categoricals columns in data.

    Attributes
    ----------
    data : pd.Dataframe
        Loaded data
    removed_vars : List[str]
        Keeping track of removed columns in data.
    response_var : str
        Name of response variable.
    categoricals : List[str]
        List of categoricals columns in data.

    """
    def __init__(self,
                 csv_dir: str,
                 response_var: str,
                 categoricals: List[str] = [],
                 del_var: List[str] = []):
        # read .csv data from directory
        self.data = pd.read_csv(csv_dir)

        # some inits
        assert response_var in self.data.columns, response_var + " not in data!"
        for c in categoricals:
            assert c in self.data.columns, c  + " not in data!"
        self.response_var = response_var
        self.features_cat = categoricals
        self.non_norm = None

        # keep track of removed cols
        self.removed_vars = []

        # delete variables in del_var
        if isinstance(del_var, str):
            del_var = [del_var]
        for d in del_var:
            self.removed_vars.append(d)
            self.data = self.data.drop(d, axis=1)

        # set number of features
        self._d = self.data.shape[1] - 1

        # set continous features
        self.features_cont = np.setdiff1d(self.data.columns, self.features_cat + [self.response_var])


    def hot_encode_categoricals(self, drop_first: bool = True):
        """Hot encode all features in self.features_cat.

        Parameters
        ----------
        drop_first : bool
            Drop first created column to avoid dummy trap.

        """
        for col in self.features_cat:
            # create dumm columns and merge to original df
            dummies = pd.get_dummies(self.data[col], prefix=col,
                                     drop_first=drop_first)
            self.data = pd.concat([self.data, dummies], axis=1)

            # remove dummified column
            self.data.drop(col, axis=1, inplace=True)

    def prepare_nn(self, drop_first: bool = True):
        """Prepare dataset for neural network by hot-encoding cat. features.

        Parameters
        ----------
        drop_first : bool
            Drop first created column to avoid dummy trap.

        """
        self.hot_encode_categoricals(drop_first=drop_first)
        self.non_norm = np.setdiff1d(self.x.columns, self.features_cont)


    @property
    def N(self):
        """Get number of samples."""
        return self.x.shape[0]

    @property
    def d(self):
        """Get number of features."""
        return self._d

    @property
    def x(self):
        """Get X."""
        return self.data.drop(self.response_var, axis=1)

    @property
    def y(self):
        """Get y."""
        return self.data[self.response_var]


class UCI():
    """Base class for UCI datasets.

    Note, first load the data from a specified path with self.get_data(). It is
    recommended to use self.prepare_run() before applying methods.

    Parameters
    ----------
    name : str
        Name of dataset.
    file_name : str
        Name of .csv file.
    response_var : str
        Name of response variable.
    categoricals : List[str]
        List of categorical columns in data.
    del_var : List[str]
        List of deleted columns in data.
    non_norm : List[str]
        List of columns that should not be standarized.

    Attributes
    ----------
    non_norm : List[str]
        List of columns that will not be standarized.
    name : str
        Name of dataset.
    file_name : str
        Name of .csv file.
    response_var : str
        Name of response variable.
    categoricals : List[str]
        List of categorical columns in data.
    del_var : List[str]
        List of deleted columns in data.

    """
    def __init__(self,
                 name: str,
                 file_name: str,
                 response_var: str,
                 categoricals: List[str] = [],
                 del_var: List[str] = [],
                 non_norm: List[str] = []):
        self.name = name
        self.file_name = file_name
        self.response_var = response_var
        self.categoricals = categoricals
        self.del_var = del_var

    def get_data(self, csv_dir: str):
        """Load data from path.

        Parameters
        ----------
        csv_dir : str
            Directory to load data from.

        """
        if self.file_name.endswith(".csv") is True:
            fn = self.file_name
        else:
            fn = self.file_name + ".csv"
        path = os.path.join(csv_dir, fn)
        self.data = UciData(path, self.response_var, self.categoricals, self.del_var)

    def prepare_run(self, drop_first: bool = True):
        """Prerpare UCI for neural network run by hot encoding categoricals.

        Parameters
        ----------
        drop_first : bool
            Drop first created column to avoid dummy trap.

        """
        # hot encode categoricalss and get columns that we do not want to normalize
        self.data.prepare_nn(drop_first=drop_first)

    @property
    def non_norm(self):
        """Get all columns that should not be normed."""
        return self.data.non_norm

    def make_train_test_split(self, *args, **kwargs) -> TrainTestSplit:
        """Generate train/test split of the UCI data.

        Returns
        -------
        TrainTestSplit
            Train test split object.
        """
        return TrainTestSplit(self.data.data, self.data.response_var,
                              non_norm=self.non_norm, *args, **kwargs)

    def make_gap_splits(self, *args, **kwargs):
        return make_gap_splits(self.data.x, self.data.features_cont)

    def make_tail_splits(self, *args, **kwargs):
        return make_tail_splits(self.data.x, self.data.features_cont)


class BostonHousing(UCI):
    dist_method = "ssv"
    dist_kwargs = {}
    def __init__(self, del_var=[]):
        super().__init__('Boston housing',
                         'boston',
                         'MEDV',
                         del_var=del_var,
                         categoricals=["CHAS"])


class ConcreteStrength(UCI):
    dist_method = "ssv"
    dist_kwargs = {}
    def __init__(self, del_var=[]):
        super().__init__('Concrete',
                         'concrete',
                         'Concrete-compressive-strength',
                         del_var=del_var,
                         categoricals=[])


class EnergyEfficiency(UCI):
    # https://archive.ics.uci.edu/ml/datasets/energy+efficiency
    # done here same same: https://rstudio-pubs-static.s3.amazonaws.com/244473_5d13955ea0fd4e5e9d376161b956e9dc.html
    dist_method = "ssv"
    dist_kwargs = {"noise": 0.0001,
                   "bound_left": 0,}
    def __init__(self, del_var=["y2"]):
        super().__init__('Energy',
                         'energy',
                         'y1',
                         del_var=del_var,
                         categoricals=["V5", "V6", "V8"])


class Kin8nm(UCI):
    dist_method = "gauss"  # no diff 'gauss' vs 'ssv', but 'gauss' faster
    dist_kwargs = {}
    def __init__(self, del_var=[]):
        super().__init__('Kin8nm',
                         'kin8nm',
                         'y',
                         del_var=del_var)


class NavalPropulsion(UCI):
    dist_method = "uniform"
    dist_kwargs = {}
    def __init__(self, del_var=["T1", "P1", "turb_decay_state"]):
        # comp_decay_state as target -> YG does the same
        super().__init__('Naval propulsion plant',
                         'naval-propulsion',
                         'comp_decay_state',
                         del_var=del_var,
                         categoricals=["lp", "v", "GTn", "Pexh"])


class PowerPlant(UCI):
    dist_method = "ssv"
    dist_kwargs = {}
    def __init__(self, del_var=[]):
        super().__init__('Power plant',
                         'power-plant',
                         'PE',
                         del_var=del_var)


class ProteinStucture(UCI):
    dist_method = "ssv"
    dist_kwargs = {}
    def __init__(self, del_var=[]):
        super().__init__('Protein',
                         'protein-structure',
                         'RMSD',
                         del_var=del_var)


class WineQualityRed(UCI):
    dist_method = "gauss"
    dist_kwargs = {}
    def __init__(self, del_var=[]):
        super().__init__('Wine quality red',
                         'wine',
                         'quality',
                         del_var=del_var,
                         categoricals=["density"])


class YachtHydrodynamics(UCI):
    dist_method = "ssv"
    dist_kwargs = {"bound_left": 0,}
    def __init__(self, del_var=[]):
        super().__init__('Yacht',
                         'yacht',
                         'residuary_resistance',
                         del_var=del_var)

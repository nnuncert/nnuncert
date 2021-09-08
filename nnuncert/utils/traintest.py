from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional
import copy

import numpy as np
import pandas as pd


def generate_random_split(x: np.ndarray,
                          ratio: float = 0.1,
                          rng: Union[None, int, np.random.Generator] = None
                          ) -> Tuple[np.ndarray, np.ndarray]:
    """Generate indices for random split.

    Parameters
    ----------
    x : np.ndarray
        Data used to perform random split.
    ratio : float
        Size of test = ratio * len(x)
    rng : Union[None, int, np.random.Generator]
        Method to perform the random split.

    Returns
    -------
    Tuple[np.ndarray, np.ndarray]
        Indices of train and test.

    """
    # make a random number generator
    rng = np.random.default_rng(rng)

    # create all indices
    l = len(x)
    x_idx = np.array(range(l))

    # determine number of test indices
    N_test = int(l*ratio)

    # make test indices
    test = rng.choice(x_idx, N_test, replace=False)

    # get train indices
    train = np.setdiff1d(x_idx, test)

    # shuffle indices
    rng.shuffle(test)
    rng.shuffle(train)

    return train, test


def make_gap_splits(df: pd.DataFrame, features: List[str]) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create 'gap splits' for all columns in df.

    A gap split for a column x_i in df sorts x_i and selects the middle 1/3 as
    test set: train--TEST--train.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
       List of train/test indices. One split for every column in df.

    """
    l = len(df)
    x_idx = np.array(range(l))

    # determine indices for test to select middle 1/3
    idx = list(range(int(l/3), int(l*2/3)))

    # loop through all columns of df and create train/test index
    test_idx = [np.array(df.sort_values(c).iloc[idx].index) for c in features]
    train_idx = [np.setdiff1d(x_idx, tid) for tid in test_idx]

    return list(zip(train_idx, test_idx))


def make_tail_splits(df: pd.DataFrame, features: List[str]
                    ) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Create 'tail splits' for all columns in df.

    A tail split for a column x_i in df sorts x_i and selects the outer 1/6's
    as test set: TEST--train--train--train--train--TEST.

    Parameters
    ----------
    df : pd.DataFrame

    Returns
    -------
    List[Tuple[np.ndarray, np.ndarray]]
       List of train/test indices. One split for every column in df.

    """
    l = len(df)
    x_idx = np.array(range(l))

    # determine indices for test to select outer 1/6's
    idx = list(range(0, int(np.ceil(l*1/6))))
    idx.extend(list(range(int(np.floor(l*5/6)), l)))

    # loop through all columns of df and create train/test index
    test_idx = [np.array(df.sort_values(c).iloc[idx].index) for c in features]
    train_idx = [np.setdiff1d(x_idx, tid) for tid in test_idx]

    return list(zip(train_idx, test_idx))


def norm(df: pd.DataFrame, ref: pd.DataFrame, non_norm: List[str] = []
         ) -> pd.DataFrame:
    """Standardize dataframe 'df' by reference dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    ref : pd.DataFrame
    non_norm : List[str]
        Column names to ignore for standardization.

    Returns
    -------
    pd.DataFrame
        Standardized dataframe.

    """
    # get scaling parameters from reference DataFrane
    ref_stats = ref.describe()

    # remove scaling for 'non-norm' columns
    ref_stats.loc["mean", non_norm] = 0
    ref_stats.loc["std", non_norm] = 1

    # scale df
    ref_stats = ref_stats.transpose()
    scaled_df = (df - ref_stats['mean']) / ref_stats['std']

    return scaled_df


class TrainTestSplit():
    """Create a train/test split for dataframe.

    Parameters
    ----------
    df : pd.DataFrame
    response_var : str
    train_id : Union[None, list, np.ndarray]
        Either given or randomly created based on ratio. Must be the same type
        as 'test_id'.
    test_id : Union[None, list, np.ndarray]
        Either given or randomly created based on ratio. Must be the same type
        as 'train_id'.
    non_norm : List[str]
        Columns in 'df' that should not be scaled.
    ratio : float
        Determines size of test set, must be in [0, 1[
    rng : Union[None, int, np.random.Generator]
        Argument to create random number generator.

    Attributes
    ----------
    data0 : pd.DataFrame
        Original input data.
    data_test : pd.DataFrame
        Scaled training data.
    data_train : pd.DataFrame
        Scaled test data.
    data : pd.DataFrame
        Scaled data.
    train_id : Union[list, np.ndarray]
        Training set indices.
    test_id : Union[list, np.ndarray]
        Test set indices.
    response_var : str
        Name of 'y' column in df.
    non_norm : List[str]
        Columns in data that are not normalized.

    """
    def __init__(self,
                 df: pd.DataFrame,
                 response_var: str,
                 train_id: Union[None, list, np.ndarray] = None,
                 test_id: Union[None, list, np.ndarray] = None,
                 non_norm: List[str] = [],
                 ratio: float = 0.1,
                 rng: Union[None, int, np.random.Generator] = None):
        assert type(train_id) == type(test_id), "Train_id and test_id must have the same type"

        # generate random split of size 'ratio' if train/test id are not given
        if train_id is None:
            train_id, test_id = generate_random_split(df[response_var],
                                                      ratio,
                                                      rng=rng)
        self.train_id = train_id
        self.test_id = test_id

        # some data handling
        self.data0 = df.copy()
        data_train = self.data0.loc[train_id]
        data_test = self.data0.loc[test_id]

        # handle columns that should not be scaled
        self.response_var = response_var
        non_norm = copy.copy(non_norm)
        if isinstance(non_norm, list) is False:
            non_norm = list(non_norm)
        if response_var not in non_norm:
            non_norm.extend([response_var])
        self.non_norm = non_norm

        # split data into train and test data
        self.data_test = norm(data_test, data_train, non_norm=self.non_norm)
        self.data_train = norm(data_train, data_train, non_norm=self.non_norm)
        self.data = pd.concat([self.data_test,
                               self.data_train], axis=0).sort_index()

    def scale_x(self, x: np.ndarray) -> np.ndarray:
        """Scale x by training data.

        Parameters
        ----------
        x : np.ndarray
            Input to be scaled.

        Returns
        -------
        np.ndarray
            Input array scaled by training data.

        """
        # reshape input array to proper dimensions and cast to df
        x = np.array(x).reshape(-1, self.x_train.shape[1])
        x = pd.DataFrame(x)
        x.columns = self.x_cols

        # get scale columns and scale input by training data
        non_norm = copy.copy(self.non_norm)
        non_norm.remove(self.response_var)
        x_norm = norm(x, self.x_train_unscaled_df, non_norm).values

        return x_norm

    @property
    def x_cols(self):
        """Get data features."""
        return self.data_train.drop(self.response_var, axis=1).columns

    @property
    def data_train_unscaled(self):
        """Get unscaled training data."""
        return self.data0.loc[self.train_id]

    @property
    def x_train_unscaled_df(self):
        """Get unscaled x_train as dataframe."""
        return self.data_train_unscaled.drop(self.response_var, axis=1)

    @property
    def x_train(self):
        """Get x_train."""
        return self.data_train.drop(self.response_var, axis=1).values

    @property
    def x_train_us(self):
        """Get unscaled x_train."""
        return self.data_train_unscaled.drop(self.response_var, axis=1).values

    @property
    def y_train(self):
        """Get y_train."""
        return self.data_train[self.response_var].values

    @property
    def y_train_us(self):
        """Get unsclaed y_train."""
        return self.data0.loc[self.train_id][self.response_var].values

    @property
    def x_test(self):
        """Get x_test."""
        return self.data_test.drop(self.response_var, axis=1).values

    @property
    def x_test_us(self):
        """Get unscaled x_test."""
        return self.data0.loc[self.test_id].drop(self.response_var, axis=1).values

    @property
    def y_test(self):
        """Get y_test."""
        return self.data_test[self.response_var].values

    @property
    def y_test_us(self):
        """Get unscaled y_test."""
        return self.data0.loc[self.test_id][self.response_var].values

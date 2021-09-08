from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import os
import json
import copy

import pandas as pd
import numpy as np
import scipy.stats as spstats

from nnuncert.models._pred_base import BasePred

class UCIRun:
    """Class to keep track of single UCI run of a model.

    Parameters
    ----------
    pred_train
        Prediction object of in-sample results.
    y_train
        Training responses.
    pred_test
        Prediction object of out-of-sample results.
    y_test
        Test responses.
    model : Union[None, str]
        Name of model.
    dataset : Union[None, str]
        Name of dataset.

    Attributes
    ----------
    rmse_train : float
    rmse_test : float
    log_score_train : float
    log_score_test : float
    log_score_99_train : float
    log_score_99_test : float
    log_score_95_train : float
    log_score_95_test : float
    crps_train : float
    crps_test : float
    picp_train : float
    picp_test : float
    mpiw_train : float
    mpiw_test : float
    model: str
    dataset: str

    """
    all_score_names = ['rmse_train', 'rmse_test',
                       'log_score_train', 'log_score_test',
                    #    'log_score_99_train', 'log_score_99_test',
                    #    'log_score_95_train', 'log_score_95_test',
                       'crps_train', 'crps_test',
                       'picp_train', 'picp_test',
                       'mpiw_train', 'mpiw_test']
    def __init__(self,
                 pred_train: BasePred,
                 y_train: np.ndarray,
                 pred_test: BasePred,
                 y_test: np.ndarray,
                 model: Optional[str] = None,
                 dataset: Optional[str] = None):
        self.model = model
        self.dataset = dataset
        self.pred_train = pred_train
        self.pred_test = pred_test
        self.y_train = y_train
        self.y_test = y_test

    @property
    def rmse_train(self):
        if hasattr(self, '_rmse_train') is False:
            self._rmse_train = self.pred_train.rmse(self.y_train)
        return self._rmse_train

    @property
    def rmse_test(self):
        if hasattr(self, '_rmse_test') is False:
            if self.has_test is False:
                return None
            self._rmse_test = self.pred_test.rmse(self.y_test)
        return self._rmse_test

    @property
    def log_score_train(self):
        if hasattr(self, '_log_score_train') is False:
            self._log_score_train = self.pred_train.log_score(self.y_train)
        return self._log_score_train

    @property
    def log_score_test(self):
        if hasattr(self, '_log_score_test') is False:
            if self.has_test is False:
                return None
            self._log_score_test = self.pred_test.log_score(self.y_test)
        return self._log_score_test

    @property
    def log_score_99_train(self):
        if hasattr(self, '_log_score_99_train') is False:
            self._log_score_99_train = self.pred_train.log_score_x(self.y_train,
                                                                   frac=0.99)
        return self._log_score_99_train

    @property
    def log_score_99_test(self):
        if hasattr(self, '_log_score_99_test') is False:
            if self.has_test is False:
                return None
            self._log_score_99_test = self.pred_test.log_score_x(self.y_test,
                                                                 frac=0.99)
        return self._log_score_99_test

    @property
    def log_score_95_train(self):
        if hasattr(self, '_log_score_95_train') is False:
            self._log_score_95_train = self.pred_train.log_score_x(self.y_train,
                                                                   frac=0.95)
        return self._log_score_95_train

    @property
    def log_score_95_test(self):
        if hasattr(self, '_log_score_95_test') is False:
            if self.has_test is False:
                return None
            self._log_score_95_test = self.pred_test.log_score_x(self.y_test,
                                                                 frac=0.95)
        return self._log_score_95_test

    @property
    def crps_train(self):
        if hasattr(self, '_crps_train') is False:
            self._crps_train = self.pred_train.crps(self.y_train)
        return self._crps_train

    @property
    def crps_test(self):
        if hasattr(self, '_crps_test') is False:
            if self.has_test is False:
                return None
            self._crps_test = self.pred_test.crps(self.y_test)
        return self._crps_test

    @property
    def picp_train(self):
        if hasattr(self, '_picp_train') is False:
            self._picp_train = self.pred_train.picp(self.y_train)
        return self._picp_train

    @property
    def picp_test(self):
        if hasattr(self, '_picp_test') is False:
            if self.has_test is False:
                return None
            self._picp_test = self.pred_test.picp(self.y_test)
        return self._picp_test

    @property
    def mpiw_train(self):
        if hasattr(self, '_mpiw_train') is False:
            self._mpiw_train = self.pred_train.mpiw
        return self._mpiw_train

    @property
    def mpiw_test(self):
        if hasattr(self, '_mpiw_test') is False:
            if self.has_test is False:
                return None
            self._mpiw_test = self.pred_test.mpiw
        return self._mpiw_test

    @property
    def has_test(self):
        """Check whether a out-of-sample prediction object is given."""
        if self.pred_test is not None:
            return True
        return False

    @classmethod
    def _from_path(cls, path: str):
        """Load results from a path."""
        # load json from path
        with open(path) as f:
            data = json.load(f)

        # make an empty result
        self = cls.__new__(cls)

        # load all scores
        for s in self.all_score_names:
            setattr(self, "_" + s, data[s])

        # load metadata
        for s in ["model", "dataset"]:
            setattr(self, s, data[s])

        self.pred_train = None
        self.y_train = None
        self.pred_test = None
        self.y_test = None

        return self

    def dump_pickle(self, path: str):
        """Save results via pickle."""
        self.set_scores_only()
        save_obj(self, path)

    def dump_json(self, path: str):
        """Dump scores as .json to 'path'."""
        # create folder
        dir_ = os.path.join(*os.path.split(path)[:-1])
        os.makedirs(dir_, exist_ok=True)

        # make dict to dump via json
        to_dump = ["model", "dataset"]
        to_dump += self.all_score_names
        d = {k: getattr(self, k) for k in to_dump}

        with open(path, "w") as f:
            json.dump(d, f, indent=2)

    def dump_txt(self, path: str):
        """Save result metrics to individual .txt files."""
        os.makedirs(path, exist_ok=True)
        to_dump = ["model", "dataset"]
        to_dump += score.all_score_names
        for s in to_dump:
            with open(os.path.join(path, s + ".txt"), "w") as f:
                print(getattr(self, s), file=f)

    def set_scores_only(self):
        """Delete memory heavy attributes."""
        # make all attributes
        [getattr(self, s) for s in self.all_score_names]

        # set memory heavy attributes to None
        [setattr(self, s, None)
         for s in ["pred_train", "y_train", "pred_test", "y_test"]]


class UCIResults(list):
    """Class that tracks multiple UCI runs."""

    def get(self,
            item: str,
            model: str,
            dataset: str,
            train: Optional[bool] = False) -> List[UCIRun]:
        """Get 'item' metric values for 'model'/'dataset'.

        Parameters
        ----------
        item : str
            Name of metric, must be in ["rmse", "crps", log_score", "picp",
                                        "mpiw"].
        model : str
        dataset : str
        train : bool
            Whether train or test results are returned.

        Returns
        -------
        List[UCIRun]
            Values of metric 'item' for all runs 'model'/'dataset'.

        """
        suff = {True: "_train",
                False: "_test"}[train]

        def get_item(obj, item):
            try:
                return getattr(obj, item)
            except AttributeError:
                return None

        ret = np.array([get_item(s, item + suff) for s in self
                        if s.model == model and s.dataset == dataset])

        return ret

    def get_preds(self,
                  dataset: str,
                  models: Optional[List[str]] = None,
                  train: Optional[bool] = False) -> dict:
        """Get prediction objects.

        Parameters
        ----------
        dataset : str
        models : Union[None, List[str]]
        train : bool
            Whether train or test results are returned.

        Returns
        -------
        dict
            Dict: {model_1: [pred_0, ...], model_2: [pred_0, ...], ...}

        """
        if models is None:
            models = self.models

        # get train or test preds
        item = {True: "pred_train",
                False: "pred_test"}[train]

        # get all preds for model and dataset
        d = {}
        for m in models:
            d[m] = [getattr(s, item) for s in self if s.model==m and s.dataset==dataset]
            if len(d[m]) == 1:
                d[m] = d[m][0]
        return d

    def make_df(self,
                dataset: str,
                train: Optional[bool] = False,
                metrics: Optional[List[str]] = ["rmse", "crps", "log_score", "picp", "mpiw"],
                remove_first_colindex: Optional[bool] = False,
                models: Optional[List[str]] = None) -> pd.DataFrame:
        """Collect all scores for 'models' in 'dataset'.

        Parameters
        ----------
        dataset : str
        train : bool
            Whether train or test results are returned.
        metrics : List[str]
            Default: ["rmse", "crps", "log_score", "picp", "mpiw"].
        remove_first_colindex : bool
            Flatten column names if single metric is used
        models : Union[None, List[str]]

        Returns
        -------
        pd.DataFrame
            Dataframe of results.

        """
        if models is None:
            models = self.models

        # make dataframe for dataset
        df = pd.DataFrame([self.get(met, model=mod, train=train, dataset=dataset)
                           for mod in models for met in metrics]).T
        df.columns = pd.MultiIndex.from_product([models, metrics])

        # flatten column names if single metric is used
        if len(metrics) == 1 and remove_first_colindex is True:
            df.columns = models

        return df

    def to_latex(self,
                 metrics: Optional[List[str]] = ["rmse", "crps", "log_score", "picp", "mpiw"],
                 train: Optional[bool] = False,
                 datasets: Optional[List[str]] = None,
                 models: Optional[List[str]] = None,
                 round_: Optional[int] = 3,
                 formatstr: Optional[str] = "{0:.3f}") -> str:
        """Create a latex table from the results.

        Parameters
        ----------
        metrics : List[str]
            Default: ["rmse", "crps", "log_score", "picp", "mpiw"].
        train : bool
            Whether train or test results are returned.
        datasets : Union[None, List[str]]
        models : Union[None, List[str]]
        round_ : int
            Round float values to 'round_' after comma.
        formatstr : str
            Control formatting of scores.

        Returns
        -------
        str
            Table to be pasted to latex.

        """
        def score_as_str_series(frame: pd.DataFrame, score: str, round_: int = 3, formatstr: str = "{0:.3f}") -> pd.Series:
            def filter_score(frame: pd.DataFrame, score: str):
                df = frame.copy().T
                df.index = df.index.swaplevel(-1, -2)
                return df.T[score]

            df = filter_score(frame, score)

            # find best method to make bold in latex
            mean = df.apply(np.nanmean).round(round_)
            sem = df.apply(spstats.sem).round(round_)
            if score == "log_score":
                boldidx = np.where(mean == max(mean))[0]
            elif score == "crps":
                boldidx = np.where(mean == min(mean))[0]
            elif score == "rmse":
                boldidx = np.where(mean == min(mean))[0]
            elif score == "picp":
                boldidx = np.where(mean == max(mean))[0]
            elif score == "mpiw":
                boldidx = []

            scorestr = mean.map(formatstr.format) + " (" + sem.map(formatstr.format) + ")"
            for b in boldidx:
                scorestr[b] = "\textbf{" + scorestr[b] + "}"
            return scorestr


        if models is None:
            models = self.models

        if datasets is None:
            datasets = self.datasets

        # get all scores as strings and best marked in bold
        df = pd.DataFrame([score_as_str_series(self.make_df(d, train=train, models=models), m, round_, formatstr) for d in datasets for m in metrics])

        # multiindex the rows as dataset/metric
        d = {"rmse" : "RMSE",
            "crps": "CRPS",
            "log_score": "LogS",
            "picp": "PICP",
            "mpiw": "MPIW",
            }
        metric_names = [d[m] for m in metrics]
        df.index = pd.MultiIndex.from_product([datasets, metric_names])

        # some column handling for headings
        df.columns = ["\thead{" + c + "}" for c in df.columns]

        # generate default latex for df
        latex = df.to_latex(escape=False, column_format="l @{\extracolsep{\\fill}} l" + "r"*len(df.columns))

        # add multirows and midrules to default latex
        latex_new = []
        for i, l in enumerate(latex.splitlines()):
            splits = l.split(" ")
            if splits[0] in datasets:
                # splits[0] = "\multirow{" + str(len(metrics)) + "}*{" + splits[0] + "}"
                splits[0] = "\\tblmultirow{" + splits[0] + "}"

                line = " ".join(splits)
                latex_new.append("\midrule")
                latex_new.append(line)
            else:
                latex_new.append(l)

        latex_new[0] = latex_new[0].replace("tabular}", "tabular*}{\\textwidth}")
        latex_new[-1] = latex_new[-1].replace("tabular", "tabular*")

        # handle escape characters and mulitple midrules
        latex_new = '\n'.join(latex_new)
        latex_new = latex_new.replace("_", "\_")
        latex_new = latex_new.replace("\midrule\n\midrule", "\midrule")

        return latex_new

    def _contains(self, model, dataset):
        """Check if contains results of model/dataset combination."""
        scores = [s for s in self if s.model==model and s.dataset==dataset]
        if len(scores) > 0:
            return True
        return False

    def dump(self, path: str):
        """Save object via pickle."""
        clone = deepcopy(self)
        [s.set_scores_only() for s in clone]
        save_obj(clone, path)

    @property
    def models(self):
        """Get names of all model where data is available."""
        return list(dict.fromkeys([s.model for s in self]).keys())

    @property
    def datasets(self):
        """Get names of all datasets where data is available."""
        return list(dict.fromkeys([s.dataset for s in self]).keys())

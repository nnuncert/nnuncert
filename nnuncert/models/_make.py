from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

from nnuncert.models._network import MakeNet
from nnuncert.models.dnnc import DNNCModel, DNNCRidge, DNNCHorseshoe, DNNCPred
from nnuncert.models.mc_dropout import DropoutTF, MCDropout, MCDropoutPred
from nnuncert.models.ensemble import Ensemble, PNNEnsemble, NLMEnsemble, EnsPredGauss
from nnuncert.models.gp import GPModel, GPPred
from nnuncert.models.nlm import NLM, NLMPred
from nnuncert.models.pbp import PBPModel, PBPPred
from nnuncert.models.pnn import PNN, PNNPred


STR2TYPE = {
    "DNNC-R" : DNNCRidge,
    "DNNC-HS" : DNNCHorseshoe,
    "MCDropout" : MCDropout,
    "MC Dropout" : MCDropout,
    "MC dropout" : MCDropout,
    "PNN" : PNN,
    "Deep emsemble" : PNNEnsemble,
    "GP" : GPModel,
    "GP-ReLU" : GPModel,
    "PNN-E" : PNNEnsemble,
    "NLM" : NLM,
    "NLM-E" : NLMEnsemble,
    "PBP" : PBPModel,
}


def make_network(model_type: Union[type, str],
                 input_shape: Tuple,
                 architecture: List[Tuple[int, str, float]],
                 *args, **kwargs) -> MakeNet:
    """Generate network with 'architecture' for given 'model_type'.

    Parameters
    ----------
    model_type : Union[type, str]
        Model to generate network for.
    input_shape : Tuple
        Shape of inputs for neural network.
    architecture : List[Tuple[int, str, float]]
        Network architecture, per hidden layer:
            [Number of hidden units, activation function in layer, dropout rate]

    Returns
    -------
    MakeNet
        Network to used as input for model initialization.

    """
    if isinstance(model_type, str):
        model_type = STR2TYPE[model_type]

    MakeNetDict = {
        DNNCModel : MakeNet.mean_only(input_shape, architecture, *args, **kwargs),
        DNNCRidge : MakeNet.mean_only(input_shape, architecture, *args, **kwargs),
        DNNCHorseshoe : MakeNet.mean_only(input_shape, architecture, *args, **kwargs),
        MCDropout : MakeNet.joint(input_shape, architecture, dropout_type=DropoutTF, *args, **kwargs),
        PNN : MakeNet.joint(input_shape, architecture, *args, **kwargs),
        PNNEnsemble : MakeNet.joint(input_shape, architecture, *args, **kwargs),
        NLM : MakeNet.joint(input_shape, architecture, *args, **kwargs),
        NLMEnsemble : MakeNet.joint(input_shape, architecture, *args, **kwargs),
        PBPModel : MakeNet.joint(input_shape, architecture, *args, **kwargs),
        GPModel : MakeNet.mean_only(input_shape, architecture, *args, **kwargs),
    }

    return MakeNetDict[model_type]


def make_model(model_type: Union[type, str],
               input_shape: Tuple,
               architecture: List[Tuple[int, str, float]],
               net_kwargs: Optional[Dict] = {},
               *args, **kwargs):
    """Initialize model with given architecture.

    Parameters
    ----------
    model_type : Union[type, str]
        Model to generate network for.
    input_shape : Tuple
        Shape of inputs for neural network.
    architecture : List[Tuple[int, str, float]]
        Network architecture, per hidden layer:
            [Number of hidden units, activation function in layer, dropout rate]
    net_kwargs : Optional[Dict]
        Arguments to be passed to MakeNet creator function.

    """
    if isinstance(model_type, str):
        model_type = STR2TYPE[model_type]

    # generate network
    net = make_network(model_type, input_shape, architecture, **net_kwargs)

    # init model
    model = model_type(net, *args, **kwargs)

    return model

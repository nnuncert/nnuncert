from nnuncert.models._network import MakeNet
from nnuncert.models._make import make_network, make_model
from nnuncert.models.dnnc import DNNCModel, DNNCPred
from nnuncert.models.mc_dropout import DropoutTF, MCDropout, MCDropoutPred
from nnuncert.models.ensemble import Ensemble, PNNEnsemble, NLMEnsemble, EnsPredGauss
from nnuncert.models.gp import GPModel, GPPred
from nnuncert.models.nlm import NLM, NLMPred
from nnuncert.models.pbp import PBPModel, PBPPred
from nnuncert.models.pnn import PNN, PNNPred


def type2name(model):
    """Get name of model."""
    if isinstance(model, DNNCModel):
        modelstr = "DNNC"
        if hasattr(model, "dnnc"):
            suffix = {"ridge": "-R",
                      "horseshoe": "-HS",}[model.dnnc.dnnc_type]
            modelstr = modelstr + suffix
        return modelstr
    elif isinstance(model, GPModel):
        return "GP-ReLU"
    elif isinstance(model, MCDropout):
        return "MC dropout"
    elif isinstance(model, PNN):
        return "PNN"
    elif isinstance(model, PNNEnsemble):
        return "PNN-E"
    elif isinstance(model, NLM):
        return "NLM"
    elif isinstance(model, NLMEnsemble):
        return "NLM-E"
    elif isinstance(model, PBPModel):
        return "PBP"
    else:
        raise ValueError("Type not recognized.")

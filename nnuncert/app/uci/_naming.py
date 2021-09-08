def type2name(model) -> str:
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

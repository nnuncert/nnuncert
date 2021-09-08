import pickle


def save_obj(obj, path: str):
    """Save some object via pickle.

    Parameters
    ----------
    obj
    path : str

    """
    with open(path, 'wb') as h:
        pickle.dump(obj, h)


def load_obj(path: str):
    """Load object from path via pickle.

    Parameters
    ----------
    path : str

    Returns
    -------
    Loaded object.

    """
    with open(path, 'rb') as h:
        return pickle.load(h)


def get_first_line(file: str) -> str:
    """Get first line of file.

    Parameters
    ----------
    file : str

    Returns
    -------
    str

    """
    with open(file) as f:
        return f.readline().split('\n')[0]


def save2txt(obj, file: str):
    """Save object as .txt file.

    Parameters
    ----------
    obj
    file : str
        Must end with '.txt'.

    """
    with open(file, "w") as f:
        print(obj, file=f)


def txt2float(file: str) -> float:
    """Load float from .txt file.

    Parameters
    ----------
    file : str

    Returns
    -------
    float
        Content of first line converted to float.

    """
    return float(get_first_line(file))


def txt2str(file: str) -> str:
    """Load first line from .txt file as string.

    Parameters
    ----------
    file : str

    Returns
    -------
    str
        Content of first line as string.

    """
    return get_first_line(file)

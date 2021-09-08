from typing import Tuple, Union, Iterable, List, Callable, Dict, Optional

import copy

import tensorflow as tf

from nnuncert.models.mc_dropout import DropoutTF


class MakeNet():
    def __init__(self,
                 input_shape: Tuple,
                 arch_mean: List[Tuple[int, str, float]],
                 use_bias_last: Optional[bool] = True,
                 var_head: Optional[bool] = False,
                 arch_common: Optional[List[Tuple[int, str, float]]] = None,
                 arch_log_var: Optional[List[Tuple[int, str, float]]] = None,
                 prefix: Optional[str] = None,
                 dropout_type: Union[tf.keras.layers.Dropout, DropoutTF] = tf.keras.layers.Dropout,
                 **kwargs):
        self.init_kwargs = kwargs
        self._set_prefix(prefix)
        self.input_shape = input_shape
        self.arch_mean = arch_mean
        self.arch_common = arch_common
        self.var_head = var_head
        self.arch_log_var = arch_log_var
        self.use_bias_last = use_bias_last
        self.dropout_type = dropout_type
        self._init()

    def _init(self):
        self.hidden_units = []
        # make inputs
        common = self.inputs = tf.keras.Input(shape=self.input_shape, name=self.prefix + "input")

        # create common layer if given
        if self.arch_common is not None:
            common = self._make_layers(inputs, self.arch_common, **self.init_kwargs)

        # make network that outputs only mean
        if self.var_head is False:
            self.outputs = self._make_layers(common, self.arch_mean, output_neurons=1, output_name=self.prefix + "mean", **self.init_kwargs)
            self.pred_var = False

        # make network that outputs mean and variance (PNN)
        else:
            self.pred_var = True
            # separated weight connections for mean / variance
            if self.arch_log_var is None:
                self.outputs = self._make_layers(common, self.arch_mean, output_neurons=2, output_name=self.prefix + "output", **self.init_kwargs)

            # same weight connection for mean and variance
            else:
                output_mean = self._make_layers(common, self.arch_mean, output_neurons=1, output_name=self.prefix + "mean", **self.init_kwargs)
                output_log_var = self._make_layers(common, self.arch_mean, output_neurons=1, output_name=self.prefix + "log_var", **self.init_kwargs)
                self.outputs = tf.keras.layers.Concatenate(name=self.prefix + "output")([output_mean, output_log_var])

    def _set_prefix(self, prefix: str):
        if prefix is None:
            self.prefix = ""
        elif prefix.endswith("_") is False:
            self.prefix = prefix + "_"
        else:
            self.prefix = prefix

    def clone_with_prefix(self, prefix: str):
        """Create a clone with 'prefix' at all layers."""
        clone = copy.deepcopy(self)
        clone._set_prefix(prefix)
        clone._init()
        return clone

    @classmethod
    def mean_only(cls,
                  input_shape: Tuple,
                  arch_mean: List[Tuple[int, str, float]],
                  **kwargs):
        """Make network that outputs mean only

        Parameters
        ----------
        input_shape : Tuple
            Description of parameter `input_shape`.
        arch_mean : List[Tuple[int, str, float]]
            Network architecture, per hidden layer:
                [Number of units, activation function in layer, dropout rate]

        Returns
        -------
        MakeNet
            Network to be passed to model.
        """
        return cls(input_shape, arch_mean, **kwargs)

    @classmethod
    def joint(cls,
              input_shape: Tuple,
              arch_mean: List[Tuple[int, str, float]],
              **kwargs):
        """Make network to output mean and variance (same weights but last).

        Parameters
        ----------
        input_shape : Tuple
            Description of parameter `input_shape`.
        arch_mean : List[Tuple[int, str, float]]
            Network architecture, per hidden layer:
                [Number of units, activation function in layer, dropout rate]

        Returns
        -------
        MakeNet
            Network to be passed to model.
        """
        return cls(input_shape, arch_mean, var_head=True, **kwargs)

    @classmethod
    def same(cls,
             input_shape: Tuple,
             arch_mean: List[Tuple[int, str, float]],
             **kwargs):
        """Make network to output mean and variance, where mean and variance use
        separated weight connections.

        Parameters
        ----------
        input_shape : Tuple
            Description of parameter `input_shape`.
        arch_mean : List[Tuple[int, str, float]]
            Network architecture, per hidden layer:
                [Number of units, activation function in layer, dropout rate]

        Returns
        -------
        MakeNet
            Network to be passed to model.

        """

        return cls(input_shape, arch_mean, var_head=True, arch_log_var=arch_mean, **kwargs)

    def _make_layers(self, inputs, hl, output_neurons=None, output_name=None, **dense_kwargs):
        x = inputs
        if isinstance(hl[0], list) is False:
            hl = [hl]
        x = self.dropout_type(hl[0][2]) (x)
        for i, (size, activation, droprate) in enumerate(hl):
            self.hidden_units.append(size)
            n = output_name + "_" + str(i-len(hl))
            x = tf.keras.layers.Dense(size, activation=activation, name=n) (x)
            x = self.dropout_type(droprate) (x)
        if output_neurons is not None:
            output = tf.keras.layers.Dense(output_neurons, activation="linear", name=output_name, use_bias=self.use_bias_last) (x)
            return output
        return x

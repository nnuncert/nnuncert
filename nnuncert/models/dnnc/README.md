Parts adapted from subplemental materical of Klein, N., D. J. Nott, and M. S. Smith (2021). "Marginally Calibrated Deep Distributional Regression".
In: *Journal of Computational and Graphical Statistics* 30.2, pp. 467-483.
Paper and MATLAB code can be found at [tandfonline](https://www.tandfonline.com/doi/abs/10.1080/10618600.2020.1807996).

Author of corresponding MATLAB code: Nadja Klein.

**Note that this code cannot be run as I had to remove the code adapted from MATLAB due to licensing.**

## Example Use (best run [notebook](../../../notebooks/DNNC_uci.ipynb) in Google Colab)
### Imports
```python
# general imports
import numpy as np
import numexpr as ne
import tensorflow as tf
import matplotlib.pyplot as plt

# thesis code
import nnuncert
from nnuncert.models import make_model, type2name
from nnuncert.utils.traintest import TrainTestSplit
from nnuncert.app.uci import UCI_DATASETS, load_uci
from nnuncert.utils.dist import Dist
```

### Prepare uci data and make KDE of response
```python
# load boston and look at data
uci = load_uci("boston")

# must be given proper directory where data .csv files are stored
uci.get_data("data/uci")

# prepare data by hot encoding categoricals
uci.prepare_run(drop_first=True)

# create train / test split (10 % test ratio)
# we standardize the categroical features to be zero mean and unit variance
# split has attributes such as 'x_train', 'x_test', 'y_train', 'y_test'
split = uci.make_train_test_split(ratio=0.1)

# estimate KDE for response
dist = Dist._from_values(uci.data.y, method=uci.dist_method, **uci.dist_kwargs)
```

### Fit the model to data
```python
# handle general settings
arch = [[50, "relu", 0]]  # list of hidden layer description (size, act. func, dropout rate)
epochs = 40
verbose = 0
learning_rate = 0.01

# get input shape from x_train
input_shape = split.x_train.shape[1]

# make model and compile
model = make_model("DNNC-R", input_shape, arch)
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
              metrics=["mae", "mse"])

# fit to x_train, y_train
# specify kwargs for DNNC by passing arguments into dnnc_kwargs
# example parameters: J = [burn-in, samples], theta, tau2start
# see nnuncert/models/dnnc/_dnnc.py for details
model.fit(split.x_train, split.y_train, epochs=epochs, verbose=verbose, dist=dist, dnnc_kwargs={})

# get predictions for training and test features
pred_train = model.make_prediction(split.x_train)
pred_test = model.make_prediction(split.x_test)
```

### Evaluate model performance and plot densities
```python
from nnuncert.app.uci import UCIRun

# get scores
scores = UCIRun(pred_train, split.y_train, pred_test, split.y_test,
                model="PNN-E", dataset="boston")

# scores has attributes:
# 'rmse_train', 'rmse_test', 'log_score_train', 'log_score_test', 'crps_train',
# 'crps_test', 'picp_train', 'picp_test', 'mpiw_train', 'mpiw_test'
print("RMSE: \t\t", scores.rmse_test, "\nLog Score: \t", scores.log_score_test)


from nnuncert.utils.indexing import index_to_rowcol

pred = pred_test
fig, ax = plt.subplots(2, 4, figsize=(14, 6))

# where to evaluate density
y0 = np.linspace(5, 50, 100)

# plot predictive densities (choose 8 randomly)
# randomly shuffled in train/test anyway
for i in range(8):
    r, c = index_to_rowcol(i, 4)
    ax_ = ax[r, c]
    ax_.plot(y0, pred.pdfi(i, y0))
```

### DNNC attributes
```python
# e.g., samples betas:
model.dnnc.betahat

# or acceptance rates for tau2:
model.dnnc.tau2accs

# predictive mean / variance
# Expected values, variance
pred_test.dens.Ey
pred_test.dens.Vary

# predicted log densities (n times n matrix similar to MATLAB)
pred_test.dens.lpy
```

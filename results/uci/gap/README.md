See *make_gap_splits* from [here](../../../nnuncert/utils/traintest.py) on how we created the gap splits.

Predictive performance
------
| *Logarithmic Score (LogS)* |
|:--:|
| ![box_LogS.png](_boxplot/box_uci_gap_test_log_score.png) |

| *Continuous Ranked Probability Score (CRPS)* |
|:--:|
| ![box_CRPS.png](_boxplot/box_uci_gap_test_crps.png) |

| *Root Mean Square Error (RMSE)* |
|:--:|
| ![box_RMSE.png](_boxplot/box_uci_gap_test_rmse.png) |

| *Prediction Interval Coverage Probability (PICP)* |
|:--:|
| ![box_PICP.png](_boxplot/box_uci_gap_test_picp.png) |


Results as .csv files
-----
Dataset | train.csv | test.csv
--- | --- | ---
boston | [boston_train.csv](boston_train.csv) | [boston_test.csv](boston_test.csv)
concrete | [concrete_train.csv](concrete_train.csv) | [concrete_test.csv](concrete_test.csv)
energy | [energy_train.csv](energy_train.csv) | [energy_test.csv](energy_test.csv)
kin8nm | [kin8nm_train.csv](kin8nm_train.csv) | [kin8nm_test.csv](kin8nm_test.csv)
powerplant | [powerplant_train.csv](powerplant_train.csv) | [powerplant_test.csv](powerplant_test.csv)
wine | [wine_train.csv](wine_train.csv) | [wine_test.csv](wine_test.csv)
yacht | [yacht_train.csv](yacht_train.csv) | [yacht_test.csv](yacht_test.csv)

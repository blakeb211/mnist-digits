# Goal
1. Model a classic benchmark dataset 
1. Reusable script design 
1. Good metadata logging

# Backlog 
1. DONE - grids of outlier digits
1. DONE - grid of digit means
1. DONE - cumulative distribution plot 
1. DONE - neural net modeling

# High level procedure
1. Tune on  150000
1. Model on 240000

# File structure 
1. Exploratory data analysis script
1. Tuning script
1. Modeling script
1. figures/
1. tests/

# Models
1. MLP
    * Variance Threshold, MinMaxScaler, Gridsearch hidden_layer_sizes, init_learning_rate 
1. RandomForest 
    * Variance Threshold, Gridsearch min_samples_leaf 

# Saved data 
| Contents           | Filetype          |
| :---:              | :---:             |
| metadata           | meta_*.json       |
| fitted estimator   |  grid_result*.pkl |
| images             |  *.png            |
| log data           |  log_*.md         |

# Future Work
1. @ROBUSTNESS Improve checking for whether files exist
1. @SIMPLIFY Simplify number of config parameters in tuning.py and model.py

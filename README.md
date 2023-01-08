# MNIST
* Data science pipeline applied to a classic machine learning benchmark dataset--MNIST handwritten digits

# Exploratory Data Analysis
### Averaged representations 
![Alt text](./figures/eda_mean_digit_grid.png "Averaged pictures of handwritten digits")

### Outliers extracted using matrix-matrix distances from the mean
![Alt text](./figures/eda_oddball_canberra_digit_grid.png "Oddballs - canberra metric")

![Alt text](./figures/eda_oddball_cityblock_digit_grid.png "Oddballs - cityblock metric")

# Hyperparameter tuning the neural network
![Alt text](./figures/gridsearch2.png "Hypertuning neural net dimensions")

# Model Results
* The confusion matrix shows accuracy prediction for each digit, and which digits are most commonly mistaken for others
* The F1 multiclass weighted score is 98.8
* 3 is most commonly mispredicted as 5 and 4 is most commonly mispredicted as 9
![Alt text](./figures/model_mlp1_confusion_matrix.png)

# Metadata 
* Scripts log all the important machine learning metadata in json format
![Alt text](./figures/screenshot_metadata_gridsearch.png "metadata")

# Usage
* Installing the pip package *python-mnist* puts the data downloading script *emnist_get_data.sh* in your python bin directory. e.g 
```ls PYTHON_BIN_DIR | grep mnist```
* Run tests with ```pytest tests/*```
* View json metadata with jq utility ```jq . models/*.json```
* Project **plan document** found in [plan.md](plan.md)

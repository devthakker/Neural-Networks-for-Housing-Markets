# NN-For-Boston-Housing-B351
## Using Neural Networks and training algorithms on the Boston Housing Market

### Team Members
* Devin Thakker
* Sahvan Patel
* Patrick Szpak

### Project Description
The goal of this project is to use neural networks to predict the house value of a given area in Boston. We will be using the Boston Housing Prices dataset from Kaggle. For this project we will be using Scikit-Learn and Tensorflow to build our models. We will be using the following models to train our data: Linear Regression, Decision Tree, and Neural Network. We will be comparing the results of each model to see which one performs the best.

### Dataset
https://www.kaggle.com/datasets/fedesoriano/the-boston-houseprice-data

### Models Folder will contain the following models
* Linear Regression
* Decision Tree

### Neural Network folder will contain the following models
* Neural Network using Tensorflow

## Code Setup
### I would recommend using a virtual environment for this project and installing the requirements.txt file. Navigate to the directory of this project folder in the terminal. Then run the following commands to create a virtual env, activate the env, and install the required libraries.

```python
python3 -m venv venv
source venv/bin/activate # for mac
venv\Scripts\activate # for windows

pip install -r requirements.txt
```

## Code Usage
### After navigating to the root directory of the project, you can navagate to the Models folder or NeuralNetwork folder to run the code

In each folder you will find the respective models in jupyter notebook files. You can run the code in each file to see the results of each model.

I have provided both the .ipynb files and .py files for each model. You can run the .py files by running the following command in the terminal while in the root directory of the project. This will run the code and print the results of each model.

Note: 
1. You will need to have the requirements.txt file installed for this to work
2. Each model may take time to run and each of them outputs a matplotlib graph. You will need to close the graph to see the next model's results. If you want to see the results of each model without running the code, you can look at the .ipynb files in each folder.


```python
python3 Models/LinearRegression.py
python3 Models/DecisionTree.py
python3 NeuralNetwork/NeuralNetwork.py
```



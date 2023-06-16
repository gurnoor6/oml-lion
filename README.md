# CoOpt: Comparing Optimizers

# Introduction
In this project, we compare the performance of the following 4 optimizers: SGD, Adam, Lion and Sophia. We test the performance on 2 tasks: image classification on CIFAR-10 dataset and regression on Concrete_Data.csv from lab05 of CS439 course at EPFL.

Team Members:
Gurnoor Singh Khurana <br>
Aayush Kumar <br>
Pradhit Canchi Rangam

## Project Structure
This repository contains the following important components

* `image_classification`: this contains models, utils and dataset for the image classification task
* `optimizers`: this contains the code for the Lion and Sophia optimizers
* `plots`: this contains a script to generate the plots
* `regression`: this contains models, utils and dataset for the regression task
* `run_classification.py`: this contains the driver script for classification task
* `run_regression.py`: this contains the driver script for regression task

## Running the code
To run the code, first make sure you have all the required dependencies as stated in `requirements.txt`. <br>
This code was tested on Python verion 3.8.1.

Running the code is straightforward.
* Classification task: `python run_classification.py`
* Regression task: `pytho =n run_regression.py`

After running, the plots are generated in the main directory. 

# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
This dataset contains financial and personal details of a Portugese bank's costumers.

We seek to predict whether a customer will subscribe to the bank term deposit or not.

We tried two approaches :

* Using a LogisticRegression model with scikit-learn, and tuning the model's hyperparameters using Azure HyperDrive.

* Using Azure AutoML to train different models and find the best one based on accuracy.

The best performing model was a Voting Ensemble found using the second approach.

## Scikit-learn Pipeline
First, we retrieve the dataset from the given url using TabularDataFactory class.

Then, we clean the data using the 'clean_data' method where we do some data preprocessing. Then, the dataset is split into 70% for training and 30% for testing.

We use sklearn's LogisticRegression Class to define the model and fit it.

Then, we define a SKLearn estimator to be later passed to the HyperDrive Config script. And a parameter sampler, which defines the hyperparameters we want to tune. In our case, we have the inverse regularization parameter (C), and the maximum number of iterations (max_iter).

We define a HyperDrive Config using the estimator, parameter sampler and an early termination policy. Then, we submit the experiment.

We use an early termiation policy to prevent experiments from running for a long time and using up resources.

After the run is completed, we found that the best model has C=0.1, max_iter=150, and Accuracy=0.9112797167425392.

![best model using hdr](images/best_model_hdr.png)

## AutoML
To use AutoML in Azure, we first define a AutoMLConfig in which we specify the task, training dataset, label column name, primary metric, max concurrent iterations, and iterations.

Azure AutoML tried different models such as : RandomForests, BoostedTrees, XGBoost, LightGBM, SGDClassifier, VotingEnsemble, etc. 

The best model was a Voting Ensemble that has Accuracy=0.9175374447606296.

![best model using automl](images/votingensemble.png)

Here are other metrics of the best model:

![metrics](images/best_model_aml_1.png)
![metrics](images/best_metrics_2.png)
![metrics](images/best_metrics_3.png)
![metrics](images/best_metrics_4.png)

And the Top Features :

![top k features](images/top_k.png)
![top k features](images/top_k_2.png)

## Pipeline comparison
Overall, there wasn't a big difference in accuracy between AutoML and HyperDrive.

However, in terms of architecture, AutoML was superior. With AutoML, we tried many different models that we could not have tried using HyperDrive in the same amount of time. With only one config, we could test various models. If we wanted to do the same thing using HyperDrive, we would define a config for each model.

## Future work
For future experiments, I would want to try different values in the AutoML approach for cross validation, and try other parameters to see if they will impact the final results.

## Proof of cluster clean up
![deleted compute](images/deleted_compute.png)

# Industrial-Copper-Modeling-Project

## Introduction
The objective of this project is to develop machine learning models for the copper industry to predict selling price and status

## Approach

First we perform an exploratory data analysis on the given data where we find out whether the given variables are continuous or categorical.

Here are the continous variables-
1. quantities
2. thickness
3. width
4. selling price

Here are the categiral variables-
1. country
2. item type
3. application
4. product_ref
5. customer
6. status

We then find the nul values in each column. Since null entries are very negligible compared to entire dataset, we can filter out only the not null values.
We then convert the datatypes of each column to desired one.
We then find out skewness of the columns and perform log transformation to remove this skewness on selling price and customer.
We then one hot encode status and itemtype to convert to numerical values.
For price extimation, we will include status in our feature space and vice versa for status prediction.
We now have our feature space ready along with target variable for both the models which we split into test and train data

## Model selection

For Price estimation, we selected below models and obtained r2 square for the test data-
1. Polynomial regression (deg-2): -3.946587386794174e+20
2. Decision tree: 0.8623354840120518
3. Decision tree with grid search CV: 0.8949033150434071 (Best hyperparameters: {'max_depth': 20, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 5})
4. Randon forest: 0.930888262279886
5. Random forest with grid search CV: 0.9309953876612916 (Best hyperparameters: {'max_depth': 35, 'max_features': 'sqrt', 'min_samples_leaf': 2, 'min_samples_split': 2, 'n_estimators': 150})

For status prediction, we selected below models and obtainned accuracy scores for test data-
1. Logisti Regression: 0.7855480954596823
2. Decision tree: 0.9123180216712092
3. Decision tree with grid search CV: 0.8996875623213455 (Best hyperparameters: {'max_depth': 30, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2})
4. Randon forest: 0.9332912318021671
5. Random forest with grid search CV: 0.9340224689224224 (Best hyperparameters: {'max_depth': 40, 'max_features': 'sqrt', 'min_samples_leaf': 1, 'min_samples_split': 2, 'n_estimators': 200})

In both the cases, random forest with grid search cv provided the best predictions.

After we have our model, we create the streamlit app where we take input from the user and provide the results.

## Running the app

Clone the repository.
Install the required libraries from requirements.txt
Run the Streamlit app using the command: streamlit run app.py
Provide inputs to get the Selling Price estimate and Status

Demo link: <Insert linkedin link>

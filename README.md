# [Residential Energy Prediction](https://github.com/nipun-goyal/Residential-Energy-Consumption-Prediction)

Contents:
- [Project Overview](#overview)
- [Exploratory Data Analysis and Data Transformation & Preprocessing](#eda-and-data-transformation--preprocessing)
- [Principal Component Analysis](#principal-component-analysis-pca)
- [Feature Selector](#feature-selector)
- [Hyperparameter Tuning & Model Comparison](#hyperparameter-tuning--model-comparison)
- [How to Run](#how-to-run)


## Overview
Every four years, [EIA](https://www.eia.gov/consumption/residential/) administers the Residential Energy Consumption Survey (RECS) to a nationally representative sample of housing units across the United States to collect energy characteristics data on the housing unit, usage patterns, and household demographics. 

This project focused on 2009 RECS survey data , extracted from [USA EIA website](https://www.eia.gov/consumption/residential/data/2009/) which represents the 13th iteration of the RECS program. First conducted in 1978, the Residential Energy Consumption Survey is a national sample survey that collects energy-related data for housing units occupied as a primary residence and the households that live in them. 2009 data were collected from 12,083 households selected at random using a complex multistage, area-probability sample design. The sample represents 113.6 million U.S. households, the Census Bureau’s statistical estimate for all occupied housing units in 2009 derived from their American Community Survey (ACS). 

With a combination of linear as well as Ensemble machine learning methods, the project objective was to:

- Describe/Explore the set of features with the strong statistical associations with target variable Annual Electricity Usage (in kWh)
- Use the selected features to predict total Consumption of Energy in residential homes

The target variable was "KWH" which stands for kilowatt-hour.

The Principal Component Analysis coupled with [FeatureSelector](https://github.com/nipun-goyal/Residential-Energy-Consumption-Prediction/blob/main/feature_selector.py) was utilized to determine the important features for modeling purpose. These important features were then fed into different linear and ensemble machine learning techniques to generate prediction

## EDA and Data Transformation & Preprocessing

#### The RECS 2009 survey data consisted of more than 900 variables collected across housing characteristics, appliances used, fuel types, annual consumption and cost of consumption. 

- In the EDA section, data dimesionality and data types from the RECS survey data were explored. Summary statistics was generated as well to check if there were any outlier values in the dataset. The features that were found to have outliers were later explored in-depth using Box Whisker and Kernel Density Estimate (KDE) plots. Below are snippets of box and KDE plots that were built. The outliers were identified and later dropped in data preprocessing section. For example, rows with KWH > 80,000 were dropped (only 1 row)

![Outlier 1](imgs/outlier1.png)
![Outlier 2](imgs/outlier2.png)

- No missing values were found in the dataset. Exploratory Data Analysis was largely done using scatter, bar, box-whisker and Median KWH plot to understand the relationship between variables. Median KWH plot is a plot showing median KWH values across different values of discrete numeric variables i.e. the variables whose values exist in a particular range or are countable in a finite amount of time. Below are the snippets of plots compiled in EDA section

![EDA 1](imgs/eda.png)
![EDA 2](imgs/eda1.png)
![EDA 3](imgs/eda2.png)

Based on the data Exploration, various transformations on data was performed such as, merging features, merging levels of discrete predictor features because of their low frequency count, removing the features with high number of Not Applicable (NA) values, removing imputation flags i.e. columns starting with 'Z', removing duplicate features, dropping unnecessary columns and lastly removing outliers (just one row with KWH > 80,000)

## Principal Component Analysis (PCA)

**Principal Component Analysis (PCA) is a dimension-reduction tool. Plugging in the whole dataset through PCA, the scree plot looks for elbow criterion (or bend) in the curve to show how many features can be used to include in the model.** 

The PCA Elbow curve below shows features that explain most of the variance (above 95%). Approximately, 200 features result in variance close to ~ 95%. The PCA analysis gives us a ballpark estimate of the number of features that explains majority of the variation in the dataset and hence can be used for data modeling.
![PCA1](imgs/pca1.png)


## Feature Selector
Post Data Transformation and Preprocessing, we had 428 features that were put through feature engineering to optimise the dimensions.

`FeatureSelector` class used five methods to identify the features to remove:
- Features with a high percentage (60%) of missing values
- Collinear (highly correlated, 98% correlation) features
- Features with zero importance after one hot encoding in a tree-based model
- Features with low importance (i.e. where cumulative importance is below the threshold of 90 %)
- Features with a single unique value
![Feature Importance0](imgs/feature_importance.png)
![Feature Importance1](imgs/feature_importance1.png)

Post feature selection, the original number of 428 features were reduced to 216 features.

## Baseline Models
Supervised learning usually starts with linear models. Hence, linear models coupled with ensemble methods were built and then trained using the defauly hyperparameters to form our baseline models. If we were to fit the model with the training set while evaluated with the test set, we obtained only a single sample point of evaluation with one test set. How can we be sure it is an accurate evaluation, rather than a value too low or too high by chance? If we have two models, and found that one model is better than another based on the evaluation, how can we know this is also not by chance?

Hence, cross validation technique was used to evaluate each model multiple times with different dataset and average score was considered for the decision to choose the final model candidate for evaluation on holdout dataset or test dataset. Cross validation uses k-fold to resample the same dataset multiple times and pretend they are different. With cross validation, as we are evaluating the model, or hyperparameter, the model has to be trained from scratch, each time, without reusing the training result from previous attempts.

![Baseline Models1](imgs/baseline_models.png)
![Baseline Models2](imgs/baseline_model_metrics.PNG)

Above, we evaluated each of the models multiple times with different dataset using cross validation process. From the dataframe and box plots above, we can see that the baseline models more or less perform similar. Let's re-run cross validation after tuning the model hyperparameters

## Hyperparameter Tuning & Model Comparison
For the purpose of fine tuning the hyperparameters, `GridSearchCV` was utlized. A cross validation fold of 5 was choosen across 6 models and results can be observed in the snippets below:

![Tuned Models1](imgs/tuned_models.png)
![Tuned Models2](imgs/tuned_model_metrics.PNG)

- The results show that the linear models, Lasso and Ridge, have identical average RMSE and r<sup>2</sup> scores
- The tree based model, Xtreme Gradient Boosting (XGB) has the best RMSE compared to all other linear and tree based models

## How to Run

- Clone or download github files into a local directory
- Install required python packages from [requirements.txt](https://github.com/nipun-goyal/Residential-Energy-Consumption-Prediction/blob/main/requirements.txt) file by creating virtual environment
- Activate the virtual environment
- Open Jupyter notebook and open 1.0-Residential-Energy-Modeling.ipynb
- Run notebook and observe results

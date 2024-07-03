
# Vehicle CO2 Emission Prediction

As concerns about environmental sustainability continue to grow, there is a pressing
need to mitigate greenhouse gas emissions, particularly from transportation sources.
Among these emissions, carbon dioxide (CO2) from vehicles contributes significantly to
air pollution and climate change. Therefore, this ML model aims to accurately predict CO2 emissions from
vehicles that will help policymakers, manufacturers, and consumers to implement
effective measures to reduce environmental impact.






## Table of Contents

- [Installation](#installation)
- [Objectives](#objectives)
- [Approach](#approach)
- [Tech Stacks](#tech-stacks)
- [Features](#features)
- [Conclusion](#conclusion)
- [Demo Video](#demo-video)
## Installation

To install this project, follow these steps:

   1. Clone the repository.
   2. Navigate to the project directory.
   3. Run ``
          npm install``
       to install dependencies.

## Objectives

The goal of this project is to develop a predictive model that can accurately estimate
CO2 emissions from vehicles based on various features such as vehicle type, engine
specifications, fuel type, and mileage. By leveraging machine learning techniques, we
aim to create models that not only predict CO2 emissions with high accuracy but also
provide insights into the factors influencing emissions.
## Approach

1. **Data Collection and Exploration:**
* Gather a comprehensive dataset containing information on vehicle attributes and corresponding CO2 emissions.
* Explore the dataset to understand its structure, identify potential challenges (e.g., missing data, outliers), and gain insights into the relationships between features and CO2 emissions.

2. **Data Preprocessing:**
* Handle missing values, outliers, and any inconsistencies in the data.
* Encode categorical variables and normalize numerical features to prepare the data for modeling.

3. **Feature Selection/Reduction:**
* Employ feature selection techniques to identify the most relevant features for predicting CO2 emissions.
* Utilize dimensionality reduction methods if necessary to reduce the complexity of the dataset while preserving important information.

4. **Model Selection and Training:**
* Select a diverse set of regression models including Multiple Linear Regression, Lasso Regression, Decision Tree Regression, Ridge Regression, and Random Forest Regression.
* Train each model using the preprocessed data and evaluate their performance using appropriate metrics.

5. **Model Evaluation and Interpretation:**
* Assess the predictive performance of each model using metrics such as Mean Squared Error, R-squared, and Mean Absolute Error.
* Interpret the coefficients or feature importances of the models to understand the impact of different factors on CO2 emissions.

6. **Prediction and Visualization:**
* Use the trained models to make predictions on new or unseen data.
* Visualize the predicted CO2 emissions alongside actual values to assess the accuracy of the models and identify any patterns or trends.
## Tech Stacks

**Frontend:** HTML, CSS

**Backend:** Flask, Python, Machine Learning

## Features

1. **Low Emission Vehicle (LEV):**
On taking details of vehicle like no. of cylinders, vehicle size, model name, fuel type and fuel consumption rate, the prediction wheather a vehicle is LEV or not is made.

2. **Accuracy:**
With a model accuracy of **97.63%** achieved by Random Forest Algorithm the predictions are made. 


## Conclusion

Provide recommendations for policymakers, manufacturers, and
consumers based on the model results, suggesting strategies for reducing emissions and promoting environmental sustainability in the transportation sector.
## Demo Video

[Click here to watch the demo video](https://drive.google.com/file/d/15yf12xcTHH6ZKLfzHkZnlWZzVHbmDh4D/view?usp=sharing)


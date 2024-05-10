from flask import Flask, render_template, request, redirect, url_for
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import r2_score, mean_squared_error


app = Flask(__name__)

@app.route('/')
def input():
    return render_template("input.html")


@app.route('/newpage', methods=['POST'])
def newpage():

    #---------------------------- MULTIPLE REGRESSION -----------------------------#
    
    # Get form inputs
    engine_size = float(request.form['engine-size'])
    cylinders = int(request.form['cylinders'])
    fuel_consumption_city = float(request.form['fuel_consumption_city'])
    fuel_consumption_hwy = float(request.form['fuel_consumption_hwy'])
    fuel_consumption_comb = float(request.form['fuel_consumption_comb'])
    fuel_consumption_mpg = float(request.form['fuel_consumption_mpg'])

    # Load the data
    data = pd.read_csv(r'C:\Users\SRUSHTI KAGE\Downloads\CO2 Emissions_Canada.csv')

    # Define features and target
    X = data[['Engine Size(L)', 'Cylinders', 'Fuel Consumption City (L/100 km)', 'Fuel Consumption Hwy (L/100 km)', 'Fuel Consumption Comb (L/100 km)', 'Fuel Consumption Comb (mpg)']]
    y = data['CO2 Emissions(g/km)']

    # Standardize the features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create and fit the linear regression model
    regressor = LinearRegression()
    regressor.fit(X_train, y_train)

    # Predict CO2 emissions for the new vehicle
    new_vehicle_features = [[engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_mpg]]
    new_vehicle_features_scaled = scaler.transform(new_vehicle_features)
    multiple_predicted_co2 = regressor.predict(new_vehicle_features_scaled)
    final_multiple_predicted_co2= multiple_predicted_co2[0]

    # Calculate R^2 score
    y_pred = regressor.predict(X_test)
    multiple_r2 = r2_score(y_test, y_pred)
    final_multiple_r2 = multiple_r2*100



    #------------------------------- LASSO REGRESSION -----------------------------------#

    lasso_regressor = Lasso(alpha=0.1)  
    lasso_regressor.fit(X_train, y_train)

    # Predict CO2 emissions for the new vehicle
    new_vehicle_features = [[engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_mpg]]
    new_vehicle_features_scaled = scaler.transform(new_vehicle_features)
    lasso_predicted_co2 = lasso_regressor.predict(new_vehicle_features_scaled)
    final_lasso_predicted_co2 = lasso_predicted_co2[0]

    # Calculate R^2 score
    y_pred = lasso_regressor.predict(X_test)
    lasso_r2 = r2_score(y_test, y_pred)
    final_lasso_r2 = lasso_r2 * 100



    #------------------------------- RIDGE REGRESSION -----------------------------------#


    # Create and fit the Ridge regression model
    ridge_regressor = Ridge(alpha=0.1) 
    ridge_regressor.fit(X_train, y_train)

    # Predict CO2 emissions for the new vehicle
    new_vehicle_features = [[engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_mpg]]
    new_vehicle_features_scaled = scaler.transform(new_vehicle_features)
    ridge_predicted_co2 = ridge_regressor.predict(new_vehicle_features_scaled)
    final_ridge_predicted_co2 = ridge_predicted_co2[0]

    # Calculate R^2 score
    y_pred = ridge_regressor.predict(X_test)
    ridge_r2 = r2_score(y_test, y_pred)
    final_ridge_r2 = ridge_r2 * 100





#------------------------------- DECISION REGRESSION -----------------------------------#



    # Create and fit the Decision Tree regression model
    decision_tree_regressor = DecisionTreeRegressor()
    decision_tree_regressor.fit(X_train, y_train)

    # Predict CO2 emissions for the new vehicle
    new_vehicle_features = [[engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_mpg]]
    new_vehicle_features_scaled = scaler.transform(new_vehicle_features)
    decision_predicted_co2 = decision_tree_regressor.predict(new_vehicle_features_scaled)
    final_decision_predicted_co2 = decision_predicted_co2[0]

    # Calculate R^2 score
    y_pred = decision_tree_regressor.predict(X_test)
    decision_r2 = r2_score(y_test, y_pred)
    final_decision_r2 = decision_r2 * 100



#------------------------------- RANDOM FOREST -----------------------------------#

    # Create and fit the Ridge regression model
    random_forest_regressor = RandomForestRegressor()
    random_forest_regressor.fit(X_train, y_train)

    # Predict CO2 emissions for the new vehicle
    new_vehicle_features = [[engine_size, cylinders, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb, fuel_consumption_mpg]]
    new_vehicle_features_scaled = scaler.transform(new_vehicle_features)
    random_predicted_co2 = random_forest_regressor.predict(new_vehicle_features_scaled)
    final_random_predicted_co2 = random_predicted_co2[0]

    # Calculate R^2 score
    y_pred = random_forest_regressor.predict(X_test)
    random_r2 = r2_score(y_test, y_pred)
    final_random_r2 = random_r2 * 100

    # Create dictionary of predicted values
    predicted = {
        'Multiple Regression': final_multiple_predicted_co2,
        'Lasso Regression': final_lasso_predicted_co2,
        'Ridge Regression': final_ridge_predicted_co2,
        'Decision Tree': final_decision_predicted_co2,
        'Random Forest': final_random_predicted_co2
    }
    
    # Create dictionary of accuray of all models
    accuracy = {
        'Multiple Regression': final_multiple_r2,
        'Lasso Regression': final_lasso_r2,
        'Ridge Regression': final_ridge_r2,
        'Decision Tree': final_decision_r2,
        'Random Forest': final_random_r2
    }
    

    return render_template("newpage.html", predicted=predicted, accuracy=accuracy)

if __name__ == '__main__':
    app.run(debug=True)
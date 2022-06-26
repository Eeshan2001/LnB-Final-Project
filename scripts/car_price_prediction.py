# Import required packages
import numpy as np
import pandas as pd
import pickle
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, VotingRegressor, ExtraTreesRegressor
from sklearn.tree import DecisionTreeRegressor
from datetime import date

# Reading the dataset
data = pd.read_csv("datasets/car_data.csv")

# Printing unique elements of the dataset
print("Unique elements in Seller_Type are", data["Seller_Type"].unique())
print("Unique elements in Fuel_Type are", data["Fuel_Type"].unique())
print("Unique elements in Transmission are", data["Transmission"].unique())
print("Unique elements in Owner are", data["Owner"].unique())
print("Unique elements in Year are", data["Year"].unique())
print("Unique elements in Car_Name are", data["Car_Name"].nunique())

# Prints Basic stats of the dataset
print(data.describe())

# Dropping the Car_Name Column
dataset = data[
    [
        "Year",
        "Selling_Price",
        "Present_Price",
        "Kms_Driven",
        "Fuel_Type",
        "Seller_Type",
        "Transmission",
        "Owner",
    ]
]

# Let's make a feature variable 'Present_Year' which has all the element values as current year.
dataset["Present_Year"] = date.today().year

# On subtracting 'Present_Year' and 'Year'
# we can make another feature variable as 'Number_of_Years_Old'
# which gives us idea about how old the car is.
dataset["Number_of_Years_Old"] = dataset["Present_Year"] - dataset["Year"]

# we can now safely drop 'Year' and 'Present_Year' columns
dataset.drop(labels=["Year", "Present_Year"], axis=1, inplace=True)

# select categorical variables from then dataset, and then implement categorical encoding for nominal variables
Fuel_Type = dataset[["Fuel_Type"]]
Fuel_Type = pd.get_dummies(Fuel_Type, drop_first=True)

Seller_Type = dataset[["Seller_Type"]]
Seller_Type = pd.get_dummies(Seller_Type, drop_first=True)

Transmission = dataset[["Transmission"]]
Transmission = pd.get_dummies(Transmission, drop_first=True)

dataset = pd.concat([dataset, Fuel_Type, Seller_Type, Transmission], axis=1)

dataset.drop(labels=["Fuel_Type", "Seller_Type", "Transmission"], axis=1, inplace=True)

# Dataset Splitting
X = dataset.iloc[:, :-1]
y = dataset.iloc[:, -1]

# To determine important features, make use of ExtraTreesRegressor
model = ExtraTreesRegressor()
model.fit(X, y)
print(model.feature_importances_)

# Dataset splitting
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values

# Splitting Train and Test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# Building Decision Tree Regressor model
dt_reg = DecisionTreeRegressor(random_state=0)
dt_reg.fit(X_train, y_train)
y_pred = dt_reg.predict(X_test)

# Training & Testing Accuracies
print("Decision Tree Score on Training set is", dt_reg.score(X_train, y_train))
print("Decision Tree Score on Test Set is", dt_reg.score(X_test, y_test))

# Model's accuracy
accuracies = cross_val_score(dt_reg, X_train, y_train, cv=5)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))

# Model's Mean absolute error
mae = mean_absolute_error(y_pred, y_test)
print("Mean Absolute Error:", mae)

# Model's Mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Model's Root Mean Squared error
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Model's R2 Score
print("The r2_score is", metrics.r2_score(y_test, y_pred))

# Random Forest Regression
rf_reg = RandomForestRegressor(
    n_estimators=400,
    min_samples_split=15,
    min_samples_leaf=2,
    max_features="auto",
    max_depth=30,
)
rf_reg.fit(X_train, y_train)
y_pred = rf_reg.predict(X_test)

# Training & Testing Accuracies
print("Random Forest Score on Training set is", rf_reg.score(X_train, y_train))
print("Random Forest Score on Test Set is", rf_reg.score(X_test, y_test))

# Model's accuracy
accuracies = cross_val_score(rf_reg, X_train, y_train, cv=5)
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))

# Model's Mean absolute error
mae = mean_absolute_error(y_pred, y_test)
print("Mean Absolute Error:", mae)

# Model's Mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Model's Root Mean Squared error
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Model's R2 Score
print("The r2_score is", metrics.r2_score(y_test, y_pred))

# Voting Regressor
vot_reg = VotingRegressor([("DecisionTree", dt_reg), ("RandomForestRegressor", rf_reg)])
vot_reg.fit(X_train, y_train)
y_pred = vot_reg.predict(X_test)

# Training & Testing Accuracies
print("Voting Regresssor Score on Training set is", vot_reg.score(X_train, y_train))
print("Voting Regresssor Score on Test Set is", vot_reg.score(X_test, y_test))

# Model's accuracy
accuracies = cross_val_score(vot_reg, X_train, y_train, cv=5)
print(accuracies)
print("Accuracy: {:.2f} %".format(accuracies.mean() * 100))
print("Standard Deviation: {:.2f} %".format(accuracies.std() * 100))

# Model's Mean absolute error
mae = mean_absolute_error(y_pred, y_test)
print("Mean Absolute Error:", mae)

# Models's Mean squared error
mse = mean_squared_error(y_test, y_pred)
print("Mean Squared Error:", mse)

# Model's Root Mean Squared error
print("RMSE:", np.sqrt(metrics.mean_squared_error(y_test, y_pred)))

# Model's R2 Score
print("The r2_score is", metrics.r2_score(y_test, y_pred))

# Saving Voting Regressor Model
pickle.dump(vot_reg, open("models/car_price_model.pkl", "wb"))
print("Model Saved Sucessfully...")

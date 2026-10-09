import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    GradientBoostingRegressor,
    GradientBoostingClassifier
)

from sklearn.metrics import (
    r2_score,
    mean_squared_error,
    mean_absolute_error,
    accuracy_score,
    f1_score,
    roc_auc_score
)

#Loading the dataset
df = pd.read_excel("sj_data_final.xlsx")

#Use years 1995-2010 inclusive from the dataset
df = df[(df["year"] >= 1995) & (df["year"] <= 2010)].copy()

#sort data by territory ad year
df = df.sort_values(["terr", "year"])

#Set habitat classification values
df["habitat"] = df["habitat"].map({"managed": 1,"natural": 0})

#Features
features = [
    "PINEVOL",
    "SPRUCEVOL",
    "TOTALVOL",
    "BIRCHVOL",
    "CONTORTAVOL",
    "AGE",
    "HEIGHT",
    "habitat",
    "slope_mean_degrees",
    "NDVI_mean"
]

#Regression targets
regression_targets = [
    "hatched",
    "fledl",
    "eggs"
]

#Classification targets
classification_targets = [
    "occupancy",
    "breeding success",
]

#To store final regression results
regression_results = []

print("\nRegression Targets")

#Looping through regression targets
for target in regression_targets:

    print("\n----------------------------")
    print("Regression Target:", target)
    print("----------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna()

    print("\nSamples used:", len(model_data))

    #Features and Target
    X = model_data[features]
    y = model_data[target]

    #Target Distrabution
    print("\nTarget values")
    print("----------------------------")
    print(y.value_counts().sort_index())

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

    #Gradient boosting regressor
    model = GradientBoostingRegressor(random_state=42)

    #Train model
    model.fit(X_train,y_train)

    #Predictions
    prediction = model.predict(X_test)

    #Metrics
    #R squared
    r2 = r2_score(y_test,prediction)

    #Root mean squared erro
    rmse = np.sqrt(mean_squared_error(y_test,prediction))

    #Mean absolute error
    mae = mean_absolute_error(y_test,prediction)

    #Results
    print("\nResults")
    print("----------------------------")
    print("R^2:",r2)
    print("RMSE:", rmse)
    print("MAE:", mae)

    #Feature importance
    importance = pd.DataFrame({"Variable": features, "Importance": model.feature_importances_})

    importance = importance.sort_values("Importance", ascending=False)

    print("\nVariable Importance")
    print("----------------------------")
    print(importance)

    #Store regression results
    regression_results.append({
        "Target": target,
        "Samples": len(model_data),
        "R^2": r2,
        "RMSE": rmse,
        "MAE": mae
    })

#Regression results table
regression_results_df = pd.DataFrame(regression_results)

print("\n----------------------------")
print("Regression Results")
print("----------------------------")
print(regression_results_df.to_string(index=False))

#Classification
classification_results = []

print("\n----------------------------")
print("Classification Targets")
print("----------------------------")

#Loop through classifcation targets
for target in classification_targets:

    print("\n----------------------------")
    print("Classification target:", target)
    print("----------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna()

    print("\nSamples used:",len(model_data))

    #features and target
    X = model_data[features]
    y = model_data[target]

    #Class distribution
    print("\nClass distribution")
    print("----------------------------")
    class_counts = (y.value_counts().sort_index())

    for class_value, count in class_counts.items():
        percentage = (count /len(y)) * 100

        print("Class",class_value, ":", count, "samples", "(", round(percentage, 1), "%)")

    #Stratified train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

    #Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=42)

    #Train model
    model.fit(X_train,y_train)

    #predictions
    prediction = model.predict(X_test)

    #Probability of class 1
    probability = model.predict_proba(X_test)[:, 1]

    #Metrics
    #Accuracy
    accuracy = accuracy_score(y_test,prediction)

    #F1 score
    f1 = f1_score(y_test,prediction,zero_division=0)

    #ROC_AUC
    roc_auc = roc_auc_score(y_test,probability)

    #Display results
    print("\nResults")
    print("----------------------------")
    print("Accuracy:", accuracy)
    print("F1:", f1)
    print("ROC-AUC:", roc_auc)

    #Feature Importance
    importance = pd.DataFrame({
        "Variable": features,
        "Importance": model.feature_importances_
    })

    importance = importance.sort_values("Importance",ascending=False)

    print("\nVariable Importance")
    print("----------------------------")
    print(importance)

#Store classification results
    classification_results.append({
        "Target": target,
        "Samples": len(model_data),
        "Accuracy": accuracy,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

classification_results_df = pd.DataFrame(classification_results)

print("\n----------------------------")
print("Classification Results")
print("----------------------------")
print(classification_results_df.to_string(index=False))

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.model_selection import train_test_split

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

#Loading the dataset
df = pd.read_excel("sj_data_final.xlsx")

#Use years 1995-2010 inclusive
df = df[(df["year"] >= 1995) & (df["year"] <= 2010)]

#Sort data by territory ad year
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

#Classification Taregets
classification_targets = ["occupancy","breeding success"]

#Regression targets
regression_targets = ["hatched","fledl","eggs"]

#To store final results
classification_results = []
regression_results = []

#Looping through classification targets
for target in classification_targets:

    print("\n----------------------------")
    print("Classification Target:", target)
    print("----------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna()

    print("\nSamples used:", len(model_data))

    #Target
    y = model_data[target].astype(int)

    #Class distribution
    print("\nClass distribution")
    print("----------------------------")
    print(y.value_counts().sort_index())

    #Features
    X = model_data[features]

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,)

    #Random Forest Classifier
    model = RandomForestClassifier(random_state=42)

    #Train model
    model.fit(X_train,y_train)

    #Predictions
    prediction = model.predict(X_test)

    #Prediction probabilities
    probability = (model.predict_proba(X_test)[:, 1])

    #Metrics
    #Accuracy
    accuracy = accuracy_score(y_test,prediction)

    #F1 score
    f1 = f1_score(y_test,prediction,zero_division=0)

    #ROC-AUC
    roc_auc = roc_auc_score(y_test,probability)

    #Print results
    print("\nResults")
    print("----------------------------")
    print("Accuracy:", accuracy)
    print("F1 Score:", f1)
    print("ROC-AUC:", roc_auc)

    #Variable importance
    importance = pd.DataFrame({
        "Variable": features,
        "Importance":
            model.feature_importances_
    })

    #Mean value of each feature
    mean_values = (model_data[features].mean())

    importance["Mean_value"] = (importance["Variable"].map(mean_values))

    #Sort importance
    importance = importance.sort_values("Importance",ascending=False)

    print("\nVariable Importance")
    print("-")
    print(importance)

    #Store classification results
    classification_results.append({
        "Target": target,
        "Samples": len(model_data),
        "Accuracy": accuracy,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

#Loop through regression targets
for target in regression_targets:
    print("\n----------------------------")
    print("Regression Target:", target)
    print("----------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna()

    print("\nSamples used:",len(model_data))

    #Features
    X = model_data[features]

    #Target
    y = model_data[target]

    #Show target value distribution
    print("\nTarget values")
    print("----------------------------")

    print(y.value_counts().sort_index())

    #Train test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

    #Random Forest Regresssor
    model = RandomForestRegressor(random_state=42)

    #Train model
    model.fit(X_train,y_train)

    #Predictions
    prediction = model.predict(X_test)

    #Metrics
    #R squared
    r2 = r2_score(y_test,prediction)

    #Mean squared error
    rmse = np.sqrt(mean_squared_error(y_test,prediction))

    #mean absolute error
    mae = mean_absolute_error(y_test,prediction)

    #Regression confusion matrix
    rounded_predictions = np.rint(prediction).astype(int)

    #Set correct range for each target: hatched 0-5, fledl 0-5, eggs 1-5
    if target in ["hatched", "fledl"]:
        score_labels = [0,1,2,3,4,5]

    elif target == "eggs":
        score_labels = [1,2,3,4,5]

    #Keep predictions within correct score range
    rounded_predictions = np.clip(
        rounded_predictions,
        min(score_labels),
        max(score_labels)
    )

    #Convert actual values to integers
    actual_values = np.array(y_test).astype(int)

    #Create confusion matrix
    regression_confusion_matrix = confusion_matrix(
        actual_values,
        rounded_predictions,
        labels=score_labels
    )
    
    #Confusion matrix heatmap
    plt.figure(figsize=(7, 5))

    sns.heatmap(
        regression_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=score_labels,
        yticklabels=score_labels
    )

    #Labels
    plt.xlabel("Predicted Value")

    plt.ylabel("Actual Value")

    #Title
    plt.title(target + " - Confusion Matrix")

    #Display graph
    plt.tight_layout()
    plt.show()

    #Variable importance
    importance = pd.DataFrame({"Variable": features, "Importance": model.feature_importances_})

    #Mean value of each feature
    mean_values = (model_data[features].mean())

    importance["Mean_value"] = (importance["Variable"].map(mean_values))

    #Sort by importance
    importance = importance.sort_values("Importance",ascending=False)

    print("\nVariable Importance")
    print("----------------------------")
    print(importance)

    #Store results
    regression_results.append({
        "Target": target,
        "Samples": len(model_data),
        "R^2":r2,
        "RMSE": rmse,
        "MAE": mae
    })

#Final classification comparison
classification_results_df = pd.DataFrame(classification_results)

print("\n----------------------------")
print("Final Classification Comparison")
print("----------------------------")
print(
    classification_results_df[
        [
            "Target",
            "Samples",
            "Accuracy",
            "F1",
            "ROC-AUC"
        ]
    ]
)

#Final regression comparison
regression_results_df = pd.DataFrame(regression_results)

print("\n----------------------------")
print("Final Regression Comparison")
print("----------------------------")
print(
    regression_results_df[
        [
            "Target",
            "Samples",
            "R^2",
            "RMSE",
            "MAE"
        ]
    ]
)

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.model_selection import train_test_split, ParameterGrid
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)
from sklearn.utils.class_weight import compute_sample_weight

#Loading the dataset
df = pd.read_excel("sj_data_final.xlsx")

#Use years 1995-2010 inclusive 
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

#Classification targets
classification_targets = [
    "occupancy",
    "breeding success",
    "hatched",
    "fledl",
    "eggs"
]

#To store final classification results
classification_results = []

print("\n")
print("Classification Targets")

#Loop through classifcation targets
for target in classification_targets:

    print("\n----------------------------")
    print("Classification target:", target)
    print("----------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna()

    #Convert regression targets to classification
    if target in ["hatched", "fledl", "eggs"]:
        model_data[target] = np.where(model_data[target] <= 3,0,1)

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
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42,stratify=y)

    #Calculate class weights
    sample_weights = compute_sample_weight(
        class_weight="balanced",
        y=y_train
    )

    #Display the weights being used
    unique_classes = np.unique(y_train)

    print("\nClass weights")
    print("----------------------------")

    for class_value in unique_classes:
        class_weight = sample_weights[y_train.to_numpy() == class_value][0]

        print("Class",class_value, ":", round(class_weight, 2))

    #Hyperparameter tuning
    tuning_X_train, tuning_X_test, tuning_y_train, tuning_y_test = train_test_split(
        X_train,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    tuning_weights = compute_sample_weight(
        class_weight="balanced",
        y=tuning_y_train
    )

    parameter_grid = {
        "learning_rate": [0.01, 0.05, 0.1],
        "n_estimators": [100, 300, 500],
        "max_depth": [2, 3, 5]
    }

    best_score = -1
    best_parameters = None

    print("\nHyperparameter tuning")
    print("----------------------------")

    for parameters in ParameterGrid(parameter_grid):

        tuning_model = GradientBoostingClassifier(
            learning_rate=parameters["learning_rate"],
            n_estimators=parameters["n_estimators"],
            max_depth=parameters["max_depth"],
            random_state=42
        )

        tuning_model.fit(
            tuning_X_train,
            tuning_y_train,
            sample_weight=tuning_weights
        )

        tuning_prediction = tuning_model.predict(
            tuning_X_test
        )

        tuning_f1 = f1_score(
            tuning_y_test,
            tuning_prediction,
            zero_division=0
        )

        if tuning_f1 > best_score:
            best_score = tuning_f1
            best_parameters = parameters

    print("Best parameters:", best_parameters)
    print("Best validation F1:", best_score)

    #Gradient Boosting Classifier
    model = GradientBoostingClassifier(
        learning_rate=best_parameters["learning_rate"],
        n_estimators=best_parameters["n_estimators"],
        max_depth=best_parameters["max_depth"],
        random_state=42
    )

    #Train model with class weights
    model.fit(X_train,y_train,sample_weight=sample_weights)

    #predictions
    prediction = model.predict(X_test)

    #Probability of class 1
    probability = model.predict_proba(X_test)[:, 1]

    #Metrics
    #Accuracy
    accuracy = accuracy_score(y_test,prediction)

    #F1 score
    f1 = f1_score(y_test,prediction,zero_division=0)

    #ROC-AUC
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

print("\n")
print("----------------------------")
print("Classification Results")
print("----------------------------")
print(classification_results_df.to_string(index=False))

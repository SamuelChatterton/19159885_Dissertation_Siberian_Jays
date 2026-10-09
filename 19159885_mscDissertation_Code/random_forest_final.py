import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import shap

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    cross_val_score,
    GridSearchCV
)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.utils.class_weight import compute_class_weight

#Load data set
df = pd.read_excel("sj_data_final.xlsx")

#Years 1995-2010 inclusive
df = df[(df["year"] >= 1995) & (df["year"] <= 2010)]

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

#Classification Taregets
classification_targets = [
    "occupancy",
    "breeding success",
    "hatched",
    "fledl",
    "eggs"
]

#Hyperparameters
parameter_grid = {
    "n_estimators": [300,500,800],
    "max_depth": [3,5,7,None],
    "min_samples_leaf": [1,2,3,5],
    "max_features": ["sqrt","log2",None]}

#Store results
classification_results = []

#Loop through targets
for target in classification_targets:

    print("\n----------------------------")
    print("Classification Target:", target)
    print("----------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna()

    #Convert regression targets to classification
    #Hatched: Split 0-3 / 4-5
    if target == "hatched":
        model_data[target] = np.where(model_data[target] <= 3,0,1)

    #Fledl: Split 0-3 / 4-5
    elif target == "fledl":
        model_data[target] = np.where(model_data[target] <= 3,0,1)

    #Eggs: Split 1-3 / 4-5
    elif target == "eggs":
        model_data[target] = np.where(model_data[target] <= 3,0,1)

    #Target
    y = model_data[target].astype(int)

    print("Samples used:",len(model_data))

    #Class distribution
    print("\nClass distribution")
    print("----------------------------")
    print(y.value_counts().sort_index())

    #Features
    X = model_data[features]

    #Calculate class weights
    classes = np.unique(y)

    weights = compute_class_weight(class_weight="balanced",classes=classes,y=y)

    class_weights = dict(zip(classes, weights))

    print("\nClass weights")
    print("------------")
    print(class_weights)

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42,stratify=y)

    #Random Forest Classifier
    model = RandomForestClassifier(random_state=42,class_weight=class_weights)

    #5 fold stratified cross validation
    stratified_cv = StratifiedKFold(n_splits=5,shuffle=True,random_state=42)

    #Hyperparameter tuning
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        scoring="roc_auc",
        cv=stratified_cv,
        n_jobs=-1
    )

    grid_search.fit(X_train,y_train)

    #Best model
    model = grid_search.best_estimator_

    print("\nBest Hyperparameters")
    print("----------------------------")
    print(grid_search.best_params_)

    #Final 5 fold stratified cross validation
    cv_accuracy = cross_val_score(
        model,
        X,
        y,
        cv=stratified_cv,
        scoring="accuracy"
    )

    cv_f1 = cross_val_score(
        model,
        X,
        y,
        cv=stratified_cv,
        scoring="f1"
    )

    cv_roc_auc = cross_val_score(
        model,
        X,
        y,
        cv=stratified_cv,
        scoring="roc_auc"
    )

    #Display individual fold results
    print("\n5-Fold Stratified Cross-Validation")
    print("-----------------------------------")

    for fold in range(5):
        print("\nFold", fold + 1)
        print("Accuracy:", cv_accuracy[fold])
        print("F1 Score:", cv_f1[fold])
        print("ROC-AUC:", cv_roc_auc[fold])

    #Mean and standard deviation
    print("\nCross-Validation Summary")
    print("----------------------------")
    print("\nMean Accuracy:", cv_accuracy.mean())
    print("Accuracy Standard Deviation:", cv_accuracy.std())
    print("\nMean F1:", cv_f1.mean())
    print("F1 Standard Deviation:", cv_f1.std())
    print("\nMean ROC-AUC:", cv_roc_auc.mean())
    print("ROC-AUC Standard Deviation:", cv_roc_auc.std())

    #Train best model
    model.fit(X_train, y_train)

    #Test set predictors
    prediction = model.predict(X_test)
    probability = (model.predict_proba(X_test)[:, 1])

    #Metrics
    #Accuracy
    accuracy = accuracy_score(y_test,prediction)

    #F1 score
    f1 = f1_score(y_test,prediction,zero_division=0)

    #ROC-AUC
    roc_auc = roc_auc_score(y_test,probability)

    print("\nResults")
    print("----------------------------")
    print("Accuracy:", accuracy)
    print("F1 Score:",f1)
    print("ROC-AUC:",roc_auc)

    #Confusion Matrix
    classification_confusion_matrix = confusion_matrix(y_test, prediction,labels=[0,1])

    print("\nConfusion Matrix")
    print("----------------------------")
    print(classification_confusion_matrix)

    #Confusion Matrix heatmap
    plt.figure(figsize=(6, 5))


    sns.heatmap(
        classification_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=[
            "Low",
            "High"
        ],
        yticklabels=[
            "Low",
            "High"
        ]
    )

    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(target + " - Confusion Matrix")
    plt.tight_layout()
    plt.show()

    #Variable importance
    importance = pd.DataFrame({
        "Variable": features,
        "Importance": model.feature_importances_
    })

    mean_values = (model_data[features].mean())

    importance["Mean_value"] = (importance["Variable"].map(mean_values))

    importance = importance.sort_values("Importance", ascending=False)

    print("\nVariable Importance")
    print("----------------------------")
    print(importance)

    #SHAP
    explainer = shap.TreeExplainer(model)

    shap_values = explainer.shap_values(X_test)

    #Calculate SHAP Importance
    if isinstance(shap_values, list):
        shap_importance_values = np.abs(shap_values[1]).mean(axis=0)
    else:
        if shap_values.ndim == 3:
            shap_importance_values = np.abs(shap_values[:, :, 1]).mean(axis=0)
        else:
            shap_importance_values = np.abs(shap_values).mean(axis=0)

    shap_importance = pd.DataFrame({
        "Variable": features,
        "SHAP Importance": shap_importance_values
    })

    shap_importance = shap_importance.sort_values(
        "SHAP Importance",
        ascending=False
    )

    print("\nSHAP Variable Importance")
    print("----------------------------")
    print(shap_importance)

    #SHAP Summary Plot
    if isinstance(shap_values, list):
        shap.summary_plot(
            shap_values[1],
            X_test,
            feature_names=features,
            show=True
        )
    else:
        if shap_values.ndim == 3:
            shap.summary_plot(
                shap_values[:, :, 1],
                X_test,
                feature_names=features,
                show=True
            )
        else:
            shap.summary_plot(
                shap_values,
                X_test,
                feature_names=features,
                show=True
            )
    #Store final results
    classification_results.append({
        "Target": target,
        "Samples": len(model_data),
        "Mean Accuracy": cv_accuracy.mean(),
        "Accuracy Std": cv_accuracy.std(),
        "Mean F1": cv_f1.mean(),
        "F1 Std": cv_f1.std(),
        "Mean ROC-AUC": cv_roc_auc.mean(),
        "ROC-AUC Std": cv_roc_auc.std()
    })

#Final classification Comparison
classification_results_df = pd.DataFrame(classification_results)

print("\n----------------------------")
print("Final Classification Comparison")
print("----------------------------")
print(
    classification_results_df[
        [
            "Target",
            "Samples",
            "Mean Accuracy",
            "Accuracy Std",
            "Mean F1",
            "F1 Std",
            "Mean ROC-AUC",
            "ROC-AUC Std"
        ]
    ]
)

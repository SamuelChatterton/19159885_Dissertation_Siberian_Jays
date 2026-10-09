import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

#Load the dataset
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
                   
#To store final results
classification_results = []

#Looping through classification targets
for target in classification_targets:

    print("\n----------------------------")
    print("Classification Target:", target)
    print("----------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna()

    #Convert regression targets to classification
    # Hatched: Split 0-3 / 4-5
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

    print("\nSamples used:", len(model_data))

    #Class distribution
    print("\nClass distribution")
    print("----------------------------")
    print(y.value_counts().sort_index())

    #Features
    X = model_data[features]

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.20,random_state=42)

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
    
    #Create confusion matrix
    classification_confusion_matrix = confusion_matrix(y_test, prediction,labels=[0,1])

    #Confusion matrix heatmap
    plt.figure(figsize=(6, 5))

    sns.heatmap(
        classification_confusion_matrix,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Lower", "High"],
        yticklabels=["Lower", "High"]
    )

    #Labels
    plt.xlabel("Predicted Class")
    plt.ylabel("Actual Class")
    plt.title(target + " - Confusion Matrix")
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
    classification_results.append({
        "Target": target,
        "Samples": len(model_data),
        "Accuracy": accuracy,
        "F1": f1,
        "ROC-AUC": roc_auc
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

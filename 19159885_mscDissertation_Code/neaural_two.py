import pandas as pd
import numpy as np
import os
import datetime
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)

#Set random seed
tf.keras.utils.set_random_seed(42)

#TensorFlow operations deterministic
tf.config.experimental.enable_op_determinism()

#Load dataset
df = pd.read_excel("sj_data_final.xlsx")

#Use years 1995-2010 inclusive 
df = df[(df["year"] >= 1995) &(df["year"] <= 2010)].copy()

#Sort data by territory and year
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

#Targets
targets = [
    "occupancy",
    "breeding success",
    "eggs",
    "hatched",
    "fledl"
]

#Store classification results
classification_results = []

#Loop through targets
for target in targets:
    print("\n----------------------------")
    print("Target:", target)
    print("----------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna().copy()

    #Convert regression targets to classification
    if target in ["eggs","hatched","fledl"]:

        model_data[target] = np.where(model_data[target] <= 3,0,1)

    #Features and target
    X = model_data[features].copy()
    y = model_data[target].copy()

    #Display class distribution
    print("\nClass distribution:")
    print(y.value_counts().sort_index())

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.30,random_state=42,stratify=y)

    #Standardise features using training data
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Build neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu"),
        tf.keras.layers.Dense(4, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])

    #Compile model
    model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    #TensorBoard
    log_dir = os.path.join("logs", target, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir,histogram_freq=1)

    #Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=15,restore_best_weights=True)

    #Train model
    model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        callbacks=[
            tensorboard_callback,
            early_stopping
        ],
        verbose=0,
        shuffle=True
    )

    #Model performance
    y_pred = model.predict(X_test_scaled,verbose=0)

    #Convert probabilities to classes
    y_pred_class = (y_pred.flatten() >= 0.5).astype(int)

    #Accuracy
    accuracy = accuracy_score(y_test,y_pred_class)

    #F1 score
    f1 = f1_score(y_test,y_pred_class)

    #ROC-AUC
    roc_auc = roc_auc_score(y_test,y_pred.flatten())

    #Store results
    classification_results.append({
        "Target": target,
        "Samples": len(model_data),
        "Accuracy": round(accuracy, 3),
        "F1 Score": round(f1, 3),
        "ROC-AUC": round(roc_auc, 3)
    })

    #Display results
    print("\nAccuracy:", round(accuracy, 3))

    print("F1 score:", round(f1, 3))

    print("ROC-AUC:", round(roc_auc, 3))

#Final classification comparison table
classification_table = pd.DataFrame(classification_results)

print("\n\n----------------------------")
print("Final Classification Comparison")
print("----------------------------")

print(classification_table.to_string(index=False))

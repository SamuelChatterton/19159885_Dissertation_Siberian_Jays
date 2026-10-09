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
    roc_auc_score,
    r2_score,
    mean_squared_error,
    mean_absolute_error
)

#Set random seed
tf.keras.utils.set_random_seed(42)

#TensorFlow operations deterministic
tf.config.experimental.enable_op_determinism()

#Load dataset
df = pd.read_excel("sj_data_final.xlsx")

#Use years 1995-2010 inclusive 
df = df[(df["year"] >= 1995) & (df["year"] <= 2010)].copy()

#Sort data by territory and year
df = df.sort_values(["terr", "year"])

#Set habitat classification values
df["habitat"] = df["habitat"].map({"managed": 1, "natural": 0})

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

#Store regression results
regression_results = []

#Loop through targets
for target in targets:

    print("\n------------------------------")
    print("Target:", target)
    print("------------------------------")

    #Remove rows with missing data
    model_data = df[features + [target]].dropna().copy()

    #Features and target
    X = model_data[features].copy()
    y = model_data[target].copy()

    #Identify classification targets
    classification = target in ["occupancy","breeding success"]

    #Class distribution
    if classification:

        print("\nClass distribution:")
        print(y.value_counts().sort_index())

    #Train test split
    if classification:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)
    else:
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)

    #Standardise features using training data
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Build neural network
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(32, activation="relu"),
        tf.keras.layers.Dense(16, activation="relu"),
        tf.keras.layers.Dense(8, activation="relu")
    ])

    #Add output layer
    if classification:
        model.add(tf.keras.layers.Dense(1,activation="sigmoid"))
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])
    else:
        model.add(tf.keras.layers.Dense(1))
        model.compile(optimizer="adam", loss="mse", metrics=["mse"])

    #TensorBoard
    log_dir = os.path.join("logs", target, datetime.datetime.now().strftime("%Y%m%d-%H%M%S"))

    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)


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
    y_pred = model.predict(X_test_scaled, verbose=0)

    if classification:
        #Convert probabilities to classes
        y_pred_class = (y_pred.flatten() >= 0.5).astype(int)

        #Accuracy
        accuracy = accuracy_score(y_test, y_pred_class)

        #F1 score
        f1 = f1_score(y_test, y_pred_class)

        #ROC-AUC
        roc_auc = roc_auc_score(y_test, y_pred.flatten())

        #Store classification results
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

    else:
        #R^2
        r2 = r2_score(y_test,y_pred.flatten())

        #RMSE
        rmse = np.sqrt(mean_squared_error(y_test,y_pred.flatten()))

        #MAE
        mae = mean_absolute_error(y_test,y_pred.flatten())

        #Store regression results
        regression_results.append({
            "Target": target,
            "Samples": len(model_data),
            "R^2": round(r2, 3),
            "RMSE": round(rmse, 3),
            "MAE": round(mae, 3)
        })

        #Display results
        print("\nR^2:", round(r2, 3))
        print("RMSE:", round(rmse, 3))
        print("MAE:", round(mae, 3))

#Final classification comparison table
classification_table = pd.DataFrame(
    classification_results
)

print("\n\n------------------------------")
print("Final Classification Comparison")
print("------------------------------")
print(classification_table.to_string(index=False))

#Final regression comparison table
regression_table = pd.DataFrame(regression_results)

print("\n\n------------------------------")
print("Final Regression Comparison")
print("------------------------------")
print(regression_table.to_string(index=False))

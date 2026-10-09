import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score
)
from datetime import datetime

#Set random seed
tf.keras.utils.set_random_seed(42)

#TensorFlow operations deterministic
tf.config.experimental.enable_op_determinism()

#Load dataset
file_path = "sj_data_final.xlsx"
df = pd.read_excel(file_path)

#Filter years
df = df[(df["year"] >= 1995) &(df["year"] <= 2010)].copy()

#Sort data
df = df.sort_values(["terr", "year"]).reset_index(drop=True)

#Convert habitat to numbers
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

#Hidden layer combinations
architectures = [
    #Hidden layer One
    [1],
    [2],
    [3],
    [4],
    [8],
    [16],

    #Hidden layer two
    [1, 1],
    [2, 2],
    [3, 2],
    [3, 3],
    [4, 2],
    [4, 3],
    [4, 4],
    [8, 2],
    [8, 4],
    [8, 8],
    [16, 4],
    [16, 8],
    [16, 16],

    #Hidden layer 3
    [2, 2, 1],
    [3, 2, 1],
    [3, 3, 2],
    [4, 2, 1],
    [4, 3, 2],
    [4, 4, 2],
    [8, 4, 2],
    [8, 8, 4],
    [16, 8, 4],
    [16, 16, 8],

    #Hidden layer three
    [2, 2, 2, 1],
    [3, 2, 2, 1],
    [3, 3, 2, 1],
    [4, 2, 2, 1],
    [4, 3, 2, 1],
    [4, 4, 2, 1],
    [4, 4, 4, 2],
    [8, 4, 2, 1],
    [8, 4, 4, 2],
    [8, 8, 4, 2],
    [16, 8, 4, 2],
    [16, 16, 8, 4]
]

#Store results
results = []

#Loop through targets
for target in targets:
    print("\n-------------------------")
    print("Target:", target)
    print("-------------------------")

    #Select features and target
    model_data = df[features + [target]].dropna().copy()

    #Convert regression targets into binary classes
    if target in ["eggs","hatched","fledl"]:
        model_data[target] = np.where(model_data[target] <= 3,0,1)

    #Input data
    X = model_data[features]

    #Target data
    y = model_data[target]

    print("Samples used:",len(model_data))

    #Train test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30,random_state=42,)

    #Standardise features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    #Create validation data for tuning
    tuning_X_train, tuning_X_val, tuning_y_train, tuning_y_val = train_test_split(
        X_train_scaled,
        y_train,
        test_size=0.20,
        random_state=42,
        stratify=y_train
    )

    #Store tuning results
    tuning_results = []

    #Loob through hyperparameter combinations
    for architecture in architectures:
        #Reset random seed for each architecture
        tf.keras.utils.set_random_seed(42)

        print("Testing architecture:",architecture)

        #Create model
        model = tf.keras.Sequential()

        #Input layer
        model.add(tf.keras.layers.Input(shape=(tuning_X_train.shape[1],)))

        #Add hidden layers
        for neurons in architecture:
            model.add(tf.keras.layers.Dense(neurons,activation="relu"))

        #Output layer
        model.add(tf.keras.layers.Dense(1,activation="sigmoid"))

        #Compile model
        model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

        #Train model
        model.fit(
            tuning_X_train,
            tuning_y_train,
            validation_data=(
                tuning_X_val,
                tuning_y_val
            ),
            epochs=100,
            batch_size=32,
            verbose=0,
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss",
                    patience=15,
                    restore_best_weights=True
                )
            ]
        )

        #Predict validation data
        validation_probability = model.predict(tuning_X_val,verbose=0).flatten()

        #Convert probabilities into classes
        validation_prediction = (validation_probability >= 0.5).astype(int)

        #Calculate validation F1 score
        validation_f1 = f1_score(tuning_y_val, validation_prediction,zero_division=0)

        #Store tuning results
        tuning_results.append({"Architecture": architecture, "F1": validation_f1})

    #Convert tuning results into dataframe
    tuning_results_df = pd.DataFrame(tuning_results)

    #Find best architecture
    best_index = tuning_results_df["F1"].idxmax()

    best_architecture = tuning_results_df.loc[best_index,"Architecture"]

    best_validation_f1 = tuning_results_df.loc[best_index, "F1"]

    print("\nBest architecture:")
    print(best_architecture)
    print("Validation F1:", round(best_validation_f1, 3))

    #Reset random seed before final model
    tf.keras.utils.set_random_seed(42)

    #Create final model
    final_model = tf.keras.Sequential()

    #Input layer
    final_model.add(tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)))

    #Add best hidden layers
    for neurons in best_architecture:
        final_model.add(tf.keras.layers.Dense(neurons, activation="relu"))

    #Output layer
    final_model.add(tf.keras.layers.Dense(1, activation="sigmoid"))

    #Compile final model
    final_model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

    #Create TensorBoard logw
    log_dir = ("logs/" + target.replace(" ", "_") + "/" + datetime.now().strftime("%Y%m%d-%H%M%S"))

    #TensorBoard
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)

    #Early stopping
    early_stopping = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=15,restore_best_weights=True)

    #Train final model
    final_model.fit(
        X_train_scaled,
        y_train,
        validation_split=0.20,
        epochs=100,
        batch_size=32,
        verbose=0,
        callbacks=[
            tensorboard_callback,
            early_stopping
        ]
    )

    #Predict test data
    y_probability = final_model.predict(X_test_scaled, verbose=0).flatten()

    #Convert probabilities into classes
    y_prediction = (y_probability >= 0.5).astype(int)

    #Accuracy
    accuracy = accuracy_score(y_test, y_prediction)

    #F1 score
    f1 = f1_score(y_test, y_prediction, zero_division=0)

    #Calculate ROC-AUC
    roc_auc = roc_auc_score(y_test, y_probability)

    #Print results
    print("\nFinal Results")
    print("----------------------------")
    print("Best Architecture:", best_architecture)
    print("Accuracy:", round(accuracy, 3))
    print("F1 Score:", round(f1, 3))
    print("ROC-AUC:", round(roc_auc, 3))

    #Store results
    results.append({
        "Target": target,
        "Samples": len(model_data),
        "Best Architecture": str(best_architecture),
        "Accuracy": accuracy,
        "F1": f1,
        "ROC-AUC": roc_auc
    })

#Display final results
results_df = pd.DataFrame(results)

print("\n\n-----------------------")
print("Final Results")
print("-----------------------")
print(results_df.to_string(index=False))

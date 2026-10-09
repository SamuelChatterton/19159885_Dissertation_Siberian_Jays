import pandas as pd
from sklearn.utils import shuffle
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns
from MulticlassPerceptron import MulticlassPerceptron
import tensorflow as tf
import datetime
import os

#Load Siberian Jay dataset
dataFile = "sj_data_final.xlsx"

#Load Siberian Jay dataset
sjData = pd.read_excel(dataFile)

#Preprocess data
featureColumns = [
    "AGE",
    "NDVI_mean",
    "PINEVOL",
    "SPRUCEVOL",
    "TOTALVOL",
    "BIRCHVOL",
    "CONTORTAVOL",
    "HEIGHT",
    "habitat",
    "slope_mean_degrees"
]

targetColumns = [
    "occupancy",
    "breeding success",
    "eggs",
    "hatched",
    "fledl"
]

#Encode categorical features
habitatMapping = {habitat: idx for idx, habitat in enumerate(sjData['habitat'].dropna().unique())}

sjData['habitat'] = sjData['habitat'].map(habitatMapping)

#One-hot encode labels
def one_hot_encode(y):
    numClasses = np.max(y) + 1
    return np.eye(numClasses)[y]

#Hyperparameters for hyperparameter tuning
epochList = [100,200,300,400,500,600,700,800,900,1000]
learningRateSizes = [0.001,0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8]
trainingSetSizes = [0.4, 0.5, 0.6, 0.7, 0.8, 0.9]

for targetColumn in targetColumns:


    print("\n-----------------------------")
    print(f"Target: {targetColumn}")
    print("-----------------------------")

    #Preprocess data
    targetData = sjData[featureColumns + [targetColumn]].dropna()

    #Convert target to integer classes
    targetData[targetColumn] = targetData[targetColumn].astype(int)

    #Shuffle dataset
    targetData = shuffle(targetData, random_state=42)

    print(f"Samples used: {len(targetData)}")

    #Record best hyperparameter values
    bestHyperparameters = {
        'learningRate': None,
        'epochs': None,
        'bestTrainingSetSize': None,
        'validationAccuracy': float('inf')
    }

    trainingErrorHistory = []
    learningRateResults = []
    traingSetSizeResults = []

    minTrainingError = {}
    minLearningError = {}

    #Hyperparameter tuning
    for learningRate in learningRateSizes:
        for epochs in epochList:
            for splitRatio in trainingSetSizes:
                splitIndex = int(len(targetData) * splitRatio)

                #Split dataset into training and test sets
                trainSet = targetData[:splitIndex]
                testSet = targetData[splitIndex:]

                #Separate features and labels for both training and test sets
                X_train = trainSet.drop(targetColumn, axis=1).values
                y_train = trainSet[targetColumn].values
                X_test = testSet.drop(targetColumn, axis=1).values
                y_test = testSet[targetColumn].values

                #Calculate mean and standard deviation for feature scaling
                mean = X_train.mean(axis=0)
                std = X_train.std(axis=0)

                std[std == 0] = 1

                #Scale features for training and test sets
                X_train_scaled = (X_train - mean) / std
                X_test_scaled = (X_test - mean) / std

                #Create a Multiclass Perceptron model
                mlp = MulticlassPerceptron(learningRate=learningRate)
                mlp.train(X_train_scaled, y_train, numOfIterations=epochs)

                #Calculate validation error
                errors = 0

                for i in range(len(X_test_scaled)):
                    actualLabel = y_test[i]
                    predictedLabel = mlp.predict(X_test_scaled[i])

                    if predictedLabel != actualLabel:
                        errors += 1

                validationAccuracy = errors / len(X_test_scaled)

                #Update best hyperparameters if validation error is lower
                if validationAccuracy < bestHyperparameters['validationAccuracy']:
                    bestHyperparameters['epochs'] = epochs
                    bestHyperparameters['learningRate'] = learningRate
                    bestHyperparameters['bestTrainingSetSize'] = splitRatio
                    bestHyperparameters['validationAccuracy'] = validationAccuracy

                #Store training error for current training set size
                if splitRatio not in minTrainingError or validationAccuracy < minTrainingError[splitRatio]:
                    minTrainingError[splitRatio] = validationAccuracy

                #Store training error for current learning rate
                if learningRate not in minLearningError or validationAccuracy < minLearningError[learningRate]:
                    minLearningError[learningRate] = validationAccuracy

                #Store training error history
                trainingErrorHistory.append(validationAccuracy)
                learningRateResults.append(learningRate)
                traingSetSizeResults.append(splitRatio)

    #Split dataset using best training set size
    splitIndex = int(len(targetData) * bestHyperparameters['bestTrainingSetSize'])

    trainSet = targetData[:splitIndex]
    testSet = targetData[splitIndex:]

    #Separate features and labels for both training and test sets
    X_train = trainSet.drop(targetColumn, axis=1).values
    y_train = trainSet[targetColumn].values
    X_test = testSet.drop(targetColumn, axis=1).values
    y_test = testSet[targetColumn].values

    #Calculate mean and standard deviation for feature scaling
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0)

    std[std == 0] = 1

    #Scale features for training and test sets
    X_train_scaled = (X_train - mean) / std
    X_test_scaled = (X_test - mean) / std

    #Create a Multiclass Perceptron model
    mlp = MulticlassPerceptron(learningRate=bestHyperparameters['learningRate'])
    mlp.train(X_train_scaled,y_train,numOfIterations=bestHyperparameters['epochs'])

    #TensorBoard logging
    log_dir = os.path.join(
        "logs",
        targetColumn,
        datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    )

    writer = tf.summary.create_file_writer(log_dir)

    with writer.as_default():
        for epoch, error in enumerate(mlp.errorHistory):
            tf.summary.scalar(
                "training_error",
                error,
                step=epoch
            )

    #One-hot encode labels for the test set
    y_test_encoded = one_hot_encode(y_test)

    y_test_labels = np.argmax(y_test_encoded, axis=1)

    y_pred = []

    #Evaluate model on test set
    for i in range(len(X_test_scaled)):
        predictedLabel = mlp.predict(X_test_scaled[i])
        y_pred.append(predictedLabel)

    cm = confusion_matrix(y_test_labels, y_pred)

    #Calculate baseline accuracy
    classCounts = np.bincount(y_test.astype(int))
    majorityClass = np.argmax(classCounts)
    baselineAccuracy = classCounts[majorityClass] / len(y_test)

    print(f"Baseline Accuracy: {baselineAccuracy}")
    print(f"Majority Class: {majorityClass}")
    
    #Calculate test accuracy
    testErrors = 0

    for i in range(len(X_test_scaled)):
        predictedLabel = y_pred[i]

        if predictedLabel != y_test_labels[i]:
            testErrors += 1

    testAccuracy = 1 - (testErrors / len(X_test_scaled))

    print(f"Best Hyperparameters:")
    print(f"Epochs: {bestHyperparameters['epochs']}")
    print(f"Learning Rate: {bestHyperparameters['learningRate']}")
    print(f"Training Set Proportion: {bestHyperparameters['bestTrainingSetSize']}")
    print(f"Validation Error: {bestHyperparameters['validationAccuracy']}")
    print(f"Test Accuracy: {testAccuracy}")

    with writer.as_default():
        tf.summary.scalar(
            "test_accuracy",
            testAccuracy,
            step=bestHyperparameters['epochs']
        )
    writer.close()

    #Create confusion matrix
    plt.figure(figsize=(10, 8))

    classLabels = sorted(targetData[targetColumn].unique())

    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classLabels, yticklabels=classLabels)

    plt.title(f'Confusion Matrix - {targetColumn}')
    plt.xlabel('Predicted Labels')
    plt.ylabel('True Labels')
    plt.show()

    fig, axs = plt.subplots(2, 2, figsize=(12, 8))

    #Plot training error against epochs
    axs[0, 0].plot(mlp.errorHistory)
    axs[0, 0].set_title('Training Error Over Epochs')
    axs[0, 0].set_xlabel('Epochs')
    axs[0, 0].set_ylabel('Error Rate')

    #Plot training error against learning rates
    uniqueLearningRates = np.unique(learningRateResults)

    meanTrainingErrors = [np.mean([error for rate, error in zip(learningRateResults, trainingErrorHistory) if rate == lr]) for lr in uniqueLearningRates]

    axs[0, 1].plot(uniqueLearningRates, meanTrainingErrors, marker='o')
    axs[0, 1].set_title('Training Error vs. Learning Rate')
    axs[0, 1].set_xlabel('Learning Rate')
    axs[0, 1].set_ylabel('Error Rate')

    #Plot training set size against error rate
    uniqueSetSizes = np.unique(traingSetSizeResults)
    minErrors = [minTrainingError[prop] for prop in uniqueSetSizes]

    axs[1, 0].plot(uniqueSetSizes, minErrors, marker='o')
    axs[1, 0].set_title('Training Set Size vs. Minimum Error Rate')
    axs[1, 0].set_xlabel('Training Set Proportion')
    axs[1, 0].set_ylabel('Error Rate')

    axs[1, 1].axis('off')

    plt.tight_layout()
    plt.show()


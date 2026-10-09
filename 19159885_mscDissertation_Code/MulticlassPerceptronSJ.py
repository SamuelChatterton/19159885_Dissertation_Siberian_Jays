import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

class MulticlassPerceptron:
    def __init__(self, learningRate):
        self.learningRate = learningRate
        self.labelWeights = {}
        self.errorHistory = []

    #Initialise weights for each label
    def initialise_weights(self, labels, featureCount):
        for label in labels:
            self.labelWeights[label] = np.zeros(featureCount)

    def train(self, X_train, y_train, numOfIterations, learningRateSchedule=None):
        # Initialise weights
        labels = np.unique(y_train) 
        featureCount = X_train.shape[1]
        self.initialise_weights(labels, featureCount)

        self.errorHistory = []

        for epoch in range(numOfIterations):
            errors = 0

            #Adjust learning rate to be optimal after each epoch
            if learningRateSchedule is not None:
                self.learningRate = learningRateSchedule(self.learningRate, epoch)

            for i in range(len(X_train)):
                actualLabel = y_train[i]
                featureRow = X_train[i]

                predicted_label = self.predict(featureRow)
                if predicted_label != actualLabel:
                    errors += 1
                    self.labelWeights[predicted_label] -= self.learningRate * featureRow
                    self.labelWeights[actualLabel] += self.learningRate * featureRow

            #Calculate error rate after each epoch
            errorRate = errors / len(X_train)
            self.errorHistory.append(errorRate)

    #Make predictions 
    def predict(self, featureRow):
        scores = {label: np.dot(featureRow, weight) for label, weight in self.labelWeights.items()}
        return max(scores, key=scores.get)

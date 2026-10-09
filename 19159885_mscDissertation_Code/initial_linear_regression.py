import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error

#Load Data
df = pd.read_excel("sj_data_final.xlsx")

#Year range
df = df[(df["year"] >= 1995) &(df["year"] <= 2010)]

#features
features = [
    "PINEVOL",
    "SPRUCEVOL",
    "TOTALVOL",
    "BIRCHVOL",
    "CONTORTAVOL",
    "AGE",
    "HEIGHT",
    "slope_mean_degrees",
    "NDVI_mean"
]

#Habitat class set
df["habitat"] = df["habitat"].map({"managed": 1, "natural": 0})

targets = ["occupancy","breeding success"]

#Loop through targets
for target in targets:

    print("\n------------------")
    print("Target:", target)
    print("--------------------")

    #Remove missing rows
    model_data = df[features + [target]].dropna()

    print("Samples used:",len(model_data))

    X = model_data[features]
    y = model_data[target]

    #Train test spli
    X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)

    #Linear regression model
    model = LinearRegression()

    #Train
    model.fit(X_train,y_train)

    #Predict
    prediction = model.predict(X_test)

    #Accuracy
    r2 = r2_score(y_test, prediction)

    print("\nModel Performance")
    print("---------------------")
    print("R^2:", r2)

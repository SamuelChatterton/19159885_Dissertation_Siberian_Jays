import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

pd.set_option("display.max_columns", None)
pd.set_option("display.width", None)

#Load data
df = pd.read_excel("sj_data_final.xlsx")

#Years 1995-2010 inclusive
df = df[(df["year"] >= 1995) &(df["year"] <= 2010)]

#Sort territory and year
df = df.sort_values(["terr", "year"])

#Habitat class set
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

#Multicolinearity Heatmap
correlation_data = df[features].dropna()
correlation_matrix = correlation_data.corr()

#Display
'''print("\nFeature Correlation Matrix")
print("--------------------------")
print(correlation_matrix)'''

#Create Heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(
    correlation_matrix,
    annot=True,
    fmt=".2f",
    cmap="coolwarm",
    vmin=-1,
    vmax=1,
    center=0,
    square=True
)

#Display heatmap
plt.title( "Multicollinearity Heatmap")
plt.tight_layout()
plt.show()

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

#Load dataset
df = pd.read_excel("sj_data_final.xlsx")

#Years 1995-2000 inclusive
df = df[(df["year"] >= 1995) &(df["year"] <= 2010)]

#Habitat class set
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

#Remove rows with missing values
plot_data = df[features].dropna()

#Create scatterplot matrix
g = sns.pairplot(
    plot_data,
    diag_kind=None,
    corner=False,
    height=0.9,
    aspect=2.2
)

#Individual Graph Fix
for ax in g.axes.flat:
    if ax is not None:
        ax.tick_params(
            axis="x",
            labelrotation=45,
            labelsize=6
        )
        ax.tick_params(
            axis="y",
            labelsize=6
        )

#Make figure wider
g.fig.set_size_inches(30, 13)

#Title
g.fig.suptitle("Relationships Between Features",y=1.01,fontsize=16)
plt.show()

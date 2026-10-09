import pandas as pd
import numpy as np
import pystac_client
import planetary_computer
import rasterio
from rasterio.warp import (
    calculate_default_transform,
    reproject,
    Resampling
)
from pyproj import Transformer

#Files
territory_file = "terr.csv"
input_file = "sj_data_with_slope_ndvi.xlsx"
output_file = "sj_data_final.xlsx"

#Load territory coordinates
territories = pd.read_csv(territory_file)

#Territory names 
territories["name"] = (territories["name"].astype(str).str.strip())

longitude = territories["X"].values
latitude = territories["Y"].values

#Create bounding box
#Add small buffer around territory points
buffer = 0.05

xmin = longitude.min() - buffer
xmax = longitude.max() + buffer
ymin = latitude.min() - buffer
ymax = latitude.max() + buffer

#Connect to Microsoft Planetary Computer
catalog = pystac_client.Client.open(
    "https://planetarycomputer.microsoft.com/api/stac/v1",
    modifier=planetary_computer.sign_inplace
)

#Search for Copernicus DEM
search = catalog.search(
    collections=["cop-dem-glo-30"],
    bbox=[
        xmin,
        ymin,
        xmax,
        ymax
    ]
)

items = search.item_collection()

if len(items) == 0:
    raise Exception(
        "No data found."
    )

#Use first DEM item
item = items[0]

#Open DEM
asset = item.assets["data"]
dem = rasterio.open(asset.href)

#Reproject DEM to UTM
target_crs = "EPSG:32634"

transform, width, height = calculate_default_transform(
    dem.crs,
    target_crs,
    dem.width,
    dem.height,
    *dem.bounds
)

#Create empty array for reprojected elevation data
elevation_projected = np.empty((height, width), dtype=np.float32)

#Reproject elevation data
reproject(
    source=rasterio.band(dem, 1),
    destination=elevation_projected,
    src_transform=dem.transform,
    src_crs=dem.crs,
    dst_transform=transform,
    dst_crs=target_crs,
    resampling=Resampling.bilinear
)

#Empty values
if dem.nodata is not None:
    elevation_projected[elevation_projected == dem.nodata] = np.nan

#Calculate slope
pixel_width = abs(transform.a)
pixel_height = abs(transform.e)

#Calculate elevation change in X direction
dz_dx = np.gradient(
    elevation_projected,
    pixel_width,
    axis=1
)

# Calculate elevation change in the Y direction
dz_dy = np.gradient(elevation_projected, pixel_height, axis=0)

#Calculate slope in radians
slope_radians = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))

#Convert radians to degrees
slope_degrees = np.degrees(slope_radians)

#Convert territory coordinates to UTM
transformer = Transformer.from_crs("EPSG:4326", target_crs, always_xy=True)

x_projected, y_projected = transformer.transform(longitude, latitude)

#Extract slope at each territory
slope_values = []

for x, y in zip(x_projected,y_projected):
    row, col = rasterio.transform.rowcol(transform, x, y)

    #Check point is inside the DEM
    if (0 <= row < slope_degrees.shape[0] and 0 <= col < slope_degrees.shape[1]):
        slope = slope_degrees[row, col]
    else:
        slope = np.nan
    slope_values.append(slope)
    
#Add calculated slope to territory data
territories["slope_mean_degrees"] = slope_values

print("\nCalculated Slope Values")
print("---------------------------")

print(territories[["name", "X", "Y", "slope_mean_degrees"]].to_string(index=False))

#Load original Excel data
df = pd.read_excel(input_file)

#Clean territories
df["terr_r"] = (df["terr_r"].astype(str).str.strip())
territories["name"] = (territories["name"].astype(str).str.strip())

#Create slope lookup
slope_lookup = territories.set_index("name")["slope_mean_degrees"]

#Match territory names and add slope values
df["slope_mean_degrees_new"] = (df["terr_r"].map(slope_lookup))

print("\nTerritory Slope Data Fetched")
print("----------------------------------")

matched = df["slope_mean_degrees_new"].notna().sum()
unmatched = df["slope_mean_degrees_new"].isna().sum()

print("Matched rows:", matched)
print("Unmatched rows:", unmatched)

#Show unmatched territory names
if unmatched > 0:
    print("\nUnmatched territory names:")
    print(
        df.loc[
            df["slope_mean_degrees_new"].isna(), "terr_r"
        ]
        .drop_duplicates()
        .to_string(index=False)
    )

#Replace old slope values
df["slope_mean_degrees"] = (
    df["slope_mean_degrees_new"]
)

#Remove temporary column
df = df.drop(
    columns=[
        "slope_mean_degrees_new"
    ]
)

#Display slope values
print("\Slope Values")
print("------------------------")
print(
    df[
        [
            "terr_r",
            "slope_mean_degrees"
        ]
    ]
    .drop_duplicates()
    .sort_values("terr_r")
    .to_string(index=False)
)

#Save new excel file
df.to_excel(output_file,index=False)

print("Saved:", output_file)

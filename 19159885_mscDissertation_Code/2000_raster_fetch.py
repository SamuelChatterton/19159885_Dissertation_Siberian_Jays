import rasterio
import pandas as pd
import glob
import os
import geopandas as gpd

from shapely.geometry import Point
from rasterio.mask import mask

#File containing territory centre points
points_file = "terr.csv"

#Folder containing SLU Skogskarta raster files
raster_folder = "/Users/samuelchatterton/Documents/MscArtificialIntelligence/Year2/Dissertation/Final Datasets/2000_66haNest_TerritoryCentrePoint"

#Output file
output_file = "terr_SLU_Skogskarta2000_460m.csv"

#Radius of territory in metres
buffer_radius = 460

#Loadterritory centre points
points = pd.read_csv(points_file)

#Check that the required coordinate columns
if not {"X", "Y"}.issubset(points.columns):
    raise ValueError("Invalid Columns")

#Create point geometries using the X and Y coordinates
geometry = [Point(xy)for xy in zip(points["X"], points["Y"])]

#Create GeoDataFrame using WGS84 coordinates
gdf = gpd.GeoDataFrame(points,geometry=geometry,crs="EPSG:4326")

#Find all raster files in the raster folder
all_rasters = glob.glob(os.path.join(raster_folder, "*.tif"))

print("Total rasters found:", len(all_rasters))

#Select the raster files needed for the analysis
selected_rasters = []
for raster in all_rasters:
    name = os.path.basename(raster)
    if "_XX_" in name and "_P_00" in name:
        selected_rasters.append(raster)

print("Selected rasters:", len(selected_rasters))

#Stop the program if no suitable rasters were found
if len(selected_rasters) == 0:raise ValueError("No matching rasters")

#Process each selected raster
for raster in selected_rasters:
    filename = os.path.basename(raster)
    print("Processing:", filename)

    with rasterio.open(raster) as src:
        raster_crs = src.crs

        #Convert territory points to raster coordinate system
        points_projected = gdf.to_crs(raster_crs)

        #Create 460m radius buffer around each territory centre
        buffers = points_projected.geometry.buffer(buffer_radius)

        values = []

        #Calculate mean raster value within each territory buffer
        for buffer in buffers:
            try:
                out_image, _ = mask(src,[buffer],crop=True)

                #Extract raster data
                data = out_image[0]

                #Remove empty values from the calculation
                if src.nodata is not None:
                    data = data[data != src.nodata]

                #Store None if no valid raster values available
                if len(data) == 0:
                    values.append(None)

                #Calculate and store mean raster value
                else:
                    values.append(float(data.mean()))

            except ValueError:
                values.append(None)

        #Remove file extension from raster name
        clean = filename.replace(".tif", "")

        #Remove unnecessary prefixes from raster name
        for prefix in [
            "ndcl_",
            "t66_",
            "ext_tr_",
            "t66_ndcl_"
        ]:
            clean = clean.replace(prefix, "")

        #Extract variable name from raster filename
        variable = clean.split("_XX_")[0]

        #Create output column name
        column_name = "2000_" + variable

        #Add mean raster values to dataset
        points[column_name] = values

#Save updated dataset
points.to_csv(output_file,index=False)

print("Saved:", output_file)

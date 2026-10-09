import rasterio
import pandas as pd
import glob
import os
import geopandas as gpd
from shapely.geometry import Point
from rasterio.mask import mask

#File containing territory centre points
points_file = "terr.csv"

#Folder containing 2005 SLU Skogskarta raster files
raster_folder = "/Users/samuelchatterton/Documents/MscArtificialIntelligence/Year2/Dissertation/Final Datasets/66haNest_2005"

#Output file
output_file = "terr_SLU_Skogskarta2005_460m.csv"

#Radius of territory buffer in metres
buffer_radius = 460

#Load territory centre points
points = pd.read_csv(points_file)

#Check required coordinate columns 
if not {"X", "Y"}.issubset(points.columns):
    raise ValueError("Invalid columns")

#Create point geometries using X and Y coordinates
geometry = [Point(xy)for xy in zip(points["X"], points["Y"])]

#Create GeoDataFrame using WGS84 coordinates
gdf = gpd.GeoDataFrame(points,geometry=geometry,crs="EPSG:4326")

#Find all raster files in raster folder
all_rasters = glob.glob(os.path.join(raster_folder, "*.tif"))

print("Total rasters found:", len(all_rasters))

#Select 2005 raster files needed 
selected_rasters = []
for raster in all_rasters:
    name = os.path.basename(raster)
    if "_XX_" in name and "_P_05" in name:
        selected_rasters.append(raster)

print("Selected rasters:", len(selected_rasters))

#Stop if no suitable rasters were found
if len(selected_rasters) == 0:
    raise ValueError("No matching rasters")

#Process each selected raster
for raster in selected_rasters:
    filename = os.path.basename(raster)
    print("Processing:", filename)
    with rasterio.open(raster) as src:
        raster_crs = src.crs

        #Convert territory points to the raster coordinate system
        points_projected = gdf.to_crs(raster_crs)

        #Create 460m buffer around each territory centre
        buffers = points_projected.geometry.buffer(buffer_radius)

        values = []

        #Calculate mean raster value within each buffer
        for buffer in buffers:
            try:
                out_image, _ = mask(src,[buffer],crop=True)

                #Extract raster data
                data = out_image[0]


                #Remove empty values
                if src.nodata is not None:
                    data = data[data != src.nodata]

                #Store None if no valid values 
                if len(data) == 0:
                    values.append(None)

                #Calculate mean raster value
                else:
                    values.append(float(data.mean()))

            except ValueError:
                values.append(None)

        #Remove file extension
        clean = filename.replace(".tif", "")

        #Remove unnecessary prefixes from filename
        for prefix in [
            "ndcl_",
            "t66_",
            "ext_tr_",
            "t66_ndcl_"
        ]:
            clean = clean.replace(prefix, "")

        #Extract variable name
        variable = clean.split("_XX_")[0]

        #Create 2005 output column name
        column_name = "2005_" + variable

        #Add mean values to dataset
        points[column_name] = values

#Save the updated dataset
points.to_csv(output_file,index=False)

print("Saved:", output_file)

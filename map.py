import geopandas as gpd
import matplotlib.pyplot as plt
gdf = gpd.read_file("Census2020_BG_Neighborhoods/Census2020_BG_Neighborhoods.shp")

#gdf = gpd.read_file("tl_2025_25_tabblock20/tl_2025_25_tabblock20.shp")
print(gdf.head(25))
gdf.plot()
plt.show()
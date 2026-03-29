import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box
import pandas as pd

# so I think a way to represent for right now at least with the geopandas could be like
# overlaying a grid on top of boston and each grid cell is one index in the chromosome.
# 1 means put a store in that area, 0 means don't.
# I'm not sure how to pick exact adresses so I think this right now is picking like a grid cell
# or a zone on the map that is 0.5 miles by 0.5 miles 
# and then we can pick a random point in that cell to be the store location and try to optimize
# which cell is gonna maximize our fitness function. Then for an actual store location
# we would have ot go to that zone and check if an an empty lot or commercial space is open. 
neighborhoods = gpd.read_file("Census2020_BG_Neighborhoods/Census2020_BG_Neighborhoods.shp")
neighborhoods3 = gpd.read_file("Census2020_BG_Neighborhoods/Census2020_BG_Neighborhoods.shp")
pop_data = pd.read_csv("Boston_Race_Ethnicity_2025.csv")
health_data = pd.read_csv("/Users/annarose/Downloads/CS4100_ProjectCode/CS4100-Project/Hypertension Percentage by Boston Neighborhood(Sheet1).csv")
income_data = pd.read_csv("Boston_Household_Income_2024.csv")

# Convert population to numeric first
pop_data["Total Population"] = pd.to_numeric(
    pop_data["Total Population"].str.replace(",", ""), errors="coerce"
)

#Convert health data to numeric
health_data["Estimate"] = pd.to_numeric(
    health_data["Estimate"].astype(str).str.replace(",", ""),
# Convert income to numeric first
income_data["Median Household Income"] = pd.to_numeric(
    income_data["Median Household Income"].str.replace("$", "").str.replace(",", ""),
    errors="coerce"
)

# Merge into neighborhoods shapefile
neighborhoods = neighborhoods.merge(
    pop_data[["Neighborhood", "Total Population"]],
    left_on="BlockGr202",
    right_on="Neighborhood",
    how="left"
)

# Merge into neighborhoods shapefile
neighborhoods = neighborhoods.merge(
    health_data[["Neighborhood", "Estimate"]],
    left_on="BlockGr202",
    right_on="Neighborhood",
    how="left"
)
neighborhoods = neighborhoods.merge(
    income_data[["Neighborhood","Median Household Income"]],
    left_on="BlockGr202",
    right_on="Neighborhood",
    how="left"
)

print(neighborhoods.head())
neighborhoods["orig_area"] = neighborhoods.geometry.area
minx, miny, maxx, maxy = neighborhoods.total_bounds
# chose this size becuase i think it estimates to around 100 cells in the gird and translates
# to like 0.5 miles by 0.5 miles.
CELLSIZE = 5000

cols = np.arange(minx, maxx, CELLSIZE)
rows = np.arange(miny, maxy, CELLSIZE)

grid_cells = []
for x in cols:
    for y in rows:
        cell = box(x, y, x + CELLSIZE, y + CELLSIZE)
        grid_cells.append(cell)

grid = gpd.GeoDataFrame(geometry=grid_cells, crs=neighborhoods.crs)

# cells should only overlap Boston
boston_boundary = neighborhoods.union_all()
grid["in_boston"] = grid.geometry.intersects(boston_boundary)
candidates = grid[grid["in_boston"]].copy().reset_index(drop=True)

# Give each cell its own index
candidates["cell_idx"] = range(len(candidates))
print(f"Chromosome length: {len(candidates)}")

# Try to place 10 stores randomly on the grid. I think 10 is good to start with can see later?
n = len(candidates)
chromosome = np.zeros(n, dtype=int)
store_indices = np.random.choice(n, size=10, replace=False)
chromosome[store_indices] = 1
candidates["has_store"] = chromosome


candidate_cells = candidates[chromosome == 1].copy()
candidate_cells["reachable_area"] = candidate_cells.geometry.buffer(CELLSIZE)   
print(neighborhoods.columns)

def income_fitness_func(candidate):
    area = gpd.GeoDataFrame(
        geometry=[candidate.geometry.buffer(CELLSIZE)],
        crs=candidates.crs
    )
    overlap = gpd.overlay(neighborhoods, area, how="intersection")
    if overlap.empty:
        return -1000
    overlap["overlap_area"] = overlap.geometry.area
    num_regions = len(overlap)

    overlap["weight"] = overlap["overlap_area"] / overlap["orig_area"]
    
    income = (overlap["Total Population"] * overlap["weight"]).sum()
        
    if income == 0:
        return -1000
    income_coeff = (1 / (income/num_regions))
    print(income_coeff)
    return income_coeff

def fitness_func(chromosome):
    candidate_cells = candidates[chromosome == 1].copy()
    total_score = 0
    for _, candidate in candidate_cells.iterrows():
        area = gpd.GeoDataFrame(
            geometry=[candidate.geometry.buffer(CELLSIZE)],
            crs=candidates.crs
        )
        overlap = gpd.overlay(neighborhoods, area, how="intersection")
        if overlap.empty:
            continue
        overlap["overlap_area"] = overlap.geometry.area


        #overlap["weight"] = overlap["overlap_area"] / overlap["orig_area"]
    
        buffer_area = area.geometry.area.iloc[0]
        overlap["weight"] = overlap["overlap_area"] / buffer_area
        population = (overlap["Total Population"] * overlap["weight"]).sum()

        health = (overlap["Estimate"] * overlap["weight"]).sum()

        print("Weights:", overlap["weight"].values) 
        print("Sum of weights:", overlap["weight"].sum())
        
        score = ((0.5 * population) + (0.3 * health) + income_fitness_func(candidate) * population)
        print(f"Candidate cell {candidate['cell_idx']}")
        print(f"Population {population:.2f})")
        print(f"Health {health:.2f}")
        print(f"Neighborhood: {overlap['BlockGr202'].tolist()}")
       
        total_score += score
    return total_score
score = fitness_func(chromosome)
print("Fitness score: ")
print(score)


## PLOTS TO SEE 
fig, axes = plt.subplots(1, 3, figsize=(22, 8))
# gonna plot the original neighborhoods with labels
ax1 = axes[0]
neighborhoods.plot(ax=ax1, edgecolor="black", linewidth=1.5, color="#dbeafe")
for col, row in neighborhoods.iterrows():
    c = row.geometry.centroid
    ax1.annotate(row["BlockGr202"], xy=(c.x, c.y), fontsize=6,
                 ha="center", va="center", fontweight="bold", color="#1e3a5f")

# Label every cell with its index for second map.
ax2 = axes[1]
neighborhoods.plot(ax=ax2, edgecolor="black", linewidth=1.5, color="#f1f5f9")
candidates.plot(ax=ax2, edgecolor="#3b82f6", linewidth=0.8,
                color="#dbeafe", alpha=0.5)

for col, row in candidates.iterrows():
    c = row.geometry.centroid
    ax2.annotate(str(row["cell_idx"]), xy=(c.x, c.y), fontsize=6,
                 ha="center", va="center", color="#1e40af", fontweight="bold")
    
# mark the cells with stores in a different color on the third map
ax3 = axes[2]
neighborhoods.plot(ax=ax3, edgecolor="black", linewidth=1.5, color="#f1f5f9")
# Empty cells
candidates[candidates["has_store"] == 0].plot(
    ax=ax3, edgecolor="#cbd5e1", linewidth=0.3,
    color="#f8fafc", alpha=0.3
)
# Store cells 
candidates[candidates["has_store"] == 1].plot(
    ax=ax3, edgecolor="#1d4ed8", linewidth=2,
    color="#3b82f6", alpha=0.8
)
plt.tight_layout()
plt.show()

# Overlay Current Grocery Store Coordinates Over Map
stores_df = pd.read_csv("Boston Grocery stores - Sheet1.csv")

stores_gdf = gpd.GeoDataFrame(
    stores_df,
    geometry=gpd.points_from_xy(stores_df["Longitude"], stores_df["Latitude"]),
    crs="EPSG:4326"
)

# the store coordinates are in lat/lon but the map is in meters. 
# this converts the stores to the same system as the map so they line up correctly.
stores_gdf = stores_gdf.to_crs(neighborhoods.crs)

fig, ax = plt.subplots(figsize=(10, 10))

neighborhoods.plot(ax=ax, edgecolor="black", linewidth=1.2, color="#dbeafe", alpha=0.5)
candidates.plot(ax=ax, edgecolor="#3b82f6", linewidth=0.5, color="#f0f9ff", alpha=0.4)
stores_gdf.plot(ax=ax, color="red", markersize=40, marker="*", label="Existing Stores", zorder=5)

ax.legend()
ax.set_title("Existing Grocery Stores in Boston")
plt.tight_layout()
plt.show()
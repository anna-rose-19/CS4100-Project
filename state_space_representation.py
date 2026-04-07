import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from shapely.geometry import box
import pandas as pd
from shapely.geometry import Polygon
import warnings
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)

N_STORES = 10
CELLSIZE = 2640
MUTATION_RATE  = 0.7
MULTISWAP_PROB = 0.15

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
health_data = pd.read_csv("Hypertension Percentage by Boston Neighborhood(Sheet1).csv")
income_data = pd.read_csv("Boston_Household_Income_2024.csv")

# Convert population to numeric first
pop_data["Total Population"] = pd.to_numeric(
    pop_data["Total Population"].str.replace(",", ""), errors="coerce"
)

#Convert health data to numeric
health_data["Estimate"] = pd.to_numeric(
    health_data["Estimate"].astype(str).str.replace(",", ""), errors="coerce"
)
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

cols = np.arange(minx, maxx, CELLSIZE)
rows = np.arange(miny, maxy, CELLSIZE)

grid_cells = []
for x in cols:
    for y in rows:
        cell = box(x, y, x + CELLSIZE, y + CELLSIZE)
        grid_cells.append(cell)

grid = gpd.GeoDataFrame(geometry=grid_cells, crs=neighborhoods.crs)
print("units=", neighborhoods.crs)

# Define Logan exclusion zone (in EPSG:4326, then convert to match neighborhoods CRS)
logan_wgs84 = Polygon([
    [
        -71.0312994,
        42.3543855
      ],
      [
        -70.9962613,
        42.3389072
      ],
      [
        -70.971185,
        42.3630112
      ],
      [
        -70.9990094,
        42.3850773
      ],
      [
        -71.0127498,
        42.3873595
      ],
      [
        -71.0292383,
        42.3756938
      ],
      [
        -71.0323299,
        42.3554004
      ]
])

logan_gdf = gpd.GeoDataFrame(geometry=[logan_wgs84], crs="EPSG:4326")
logan_gdf = logan_gdf.to_crs(neighborhoods.crs)  # converts to ftUS, will line up correctly
logan_polygon = logan_gdf.geometry.iloc[0]
logan_polygon = logan_polygon.buffer(0)

# cells should only overlap Boston but exclude airport
boston_boundary = neighborhoods.union_all()
valid_area = boston_boundary.difference(logan_polygon)
grid["in_boston"] = grid.geometry.intersects(valid_area)
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

# This will check how many people are already being served by an existing grocery store. 
# Function will find which existing stores are close to the given candidate cell.
# Does this measuring the distance from each of the 59 existing stores
# to the center of the candidate cell. if a store is within CELLSIZE, it counts as nearby_store.
# Then for each nearby store, figure out how many people it already serves. add all of that up into already_served.

stores_df = pd.read_csv("Boston Grocery stores - Sheet1.csv")
stores_gdf = gpd.GeoDataFrame(
    stores_df,
    geometry=gpd.points_from_xy(stores_df["Longitude"], stores_df["Latitude"]),
    crs="EPSG:4326"
)
# the store coordinates are in lat/lon but the map is in meters. 
# this converts the stores to the same system as the map so they line up correctly.
stores_gdf = stores_gdf.to_crs(neighborhoods.crs)

def penalize_existing_stores(candidate, neighborhoods, existing_stores_gdf, CELLSIZE):
    candidate_center = candidate.geometry.centroid
    nearby_stores = existing_stores_gdf[
        existing_stores_gdf.geometry.distance(candidate_center) <= CELLSIZE
    ]
    already_served = 0
    for _, store in nearby_stores.iterrows():
        store_area = gpd.GeoDataFrame(
            geometry=[store.geometry.buffer(CELLSIZE)],
            crs=existing_stores_gdf.crs
        )
        store_overlap = gpd.overlay(neighborhoods, store_area, how="intersection")
        if store_overlap.empty:
            continue
        store_overlap["overlap_area"] = store_overlap.geometry.area
        store_overlap["weight"] = store_overlap["overlap_area"] / store_overlap["orig_area"]
        already_served += (store_overlap["Total Population"] * store_overlap["weight"]).sum()

    return already_served

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
    
    income = (overlap["Median Household Income"] * overlap["weight"]).sum()
        
    if income == 0:
        return -1000
    income_coeff = (1 / (income/num_regions))
    #print(income_coeff)
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

        already_served = penalize_existing_stores(candidate, neighborhoods, stores_gdf, CELLSIZE)

        # gives what percentage of the population in the candidate cell is already being served by an existing store.
        # for example if the candidate cell has 10,000 people in it and the nearby stores 
        # are already serving 7,000 of those people, then the coverage ratio would be 0.7.
        coverage_ratio = min(already_served / population if population > 0 else 1, 1)

        health = (overlap["Estimate"] * overlap["weight"]).sum()

        #print("Weights:", overlap["weight"].values) 
        #print("Sum of weights:", overlap["weight"].sum())

        # so this would be (0.5 * 10000)(1 - 0.7) = 1,500
        #  So placing a store here only gives you 30% of the reward you'd get in a completely uncovered area.
        score = ((0.5 * population)*(1 - coverage_ratio) + (0.3 * health) + income_fitness_func(candidate) * population)
        #print(f"Candidate cell {candidate['cell_idx']}")
        #print(f"Population {population:.2f})")
        #print(f"Health {health:.2f}")
        #print(f"Neighborhood: {overlap['BlockGr202'].tolist()}")
       
        total_score += score
    return total_score

if __name__ == "__main__":
    score = fitness_func(chromosome)
    print("Fitness score: ")
    print(score)

# pop is all generated chromosomes, fitness is an array of their scores, 
# and k is how many we are comapring for the tournament selection.
# we will run this 4? times to select parents for next gen --> not most greedy solution but prevents local optima
def selection(pop, fitness, k): 
    competitors = np.random.choice(len(pop), size=k, replace=False) #select k random chromosomes to comapare
    winner = competitors[0]
    for i in competitors:
        if fitness[i] > fitness[winner]:
            winner = i
    return pop[winner]

def mutate_swap(chromosome, n_swaps=1):
    """
    Move n_swaps stores to randomly chosen empty cells.
    Always preserves exactly N_STORES stores in the chromosome.
    """
    mutant = chromosome.copy()
    store_idxs = np.where(mutant == 1)[0]
    empty_idxs = np.where(mutant == 0)[0]

    n_swaps = min(n_swaps, len(store_idxs), len(empty_idxs))

    removes = np.random.choice(store_idxs, size=n_swaps, replace=False)
    adds    = np.random.choice(empty_idxs,  size=n_swaps, replace=False)

    mutant[removes] = 0
    mutant[adds]    = 1

    return mutant

def mutate(chromosome):
    if np.random.rand() > MUTATION_RATE:
        return chromosome.copy()           # no mutation

    if np.random.rand() < MULTISWAP_PROB:
        return mutate_swap(chromosome, n_swaps=3)  # big jump
    
    return mutate_swap(chromosome, n_swaps=1)      # small move

def new_generation(pop, fitness, num_children):
    new_pop = []
    init = num_children / 2 - 1
    for _ in range(init):
        parent = selection(pop, fitness, k=4)
        new_pop.append(parent)
        child = mutate(parent)
        new_pop.append(child)
    rand = num_children - len(new_pop)
    #for _ in range(rand):
        #randomly generate new chromosome to maintain diversity
    return np.array(new_pop)

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
fig, ax = plt.subplots(figsize=(10, 10))

neighborhoods.plot(ax=ax, edgecolor="black", linewidth=1.2, color="#dbeafe", alpha=0.5)
candidates.plot(ax=ax, edgecolor="#3b82f6", linewidth=0.5, color="#f0f9ff", alpha=0.4)
stores_gdf.plot(ax=ax, color="red", markersize=40, marker="*", label="Existing Stores", zorder=5)

ax.legend()
ax.set_title("Existing Grocery Stores in Boston")
plt.tight_layout()
plt.show()



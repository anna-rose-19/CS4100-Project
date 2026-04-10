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

N_STORES       = 10
CELLSIZE = 3500
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
    if nearby_stores.empty:
        return 0

    # union all nearby store buffers into one shape so no double-counting
    combined_area = nearby_stores.geometry.buffer(CELLSIZE).union_all()
    combined_gdf = gpd.GeoDataFrame(geometry=[combined_area], crs=existing_stores_gdf.crs)
    store_overlap = gpd.overlay(neighborhoods, combined_gdf, how="intersection")
    if store_overlap.empty:
        return 0

    store_overlap["overlap_area"] = store_overlap.geometry.area
    store_overlap["weight"] = store_overlap["overlap_area"] / store_overlap["orig_area"]
    return (store_overlap["Total Population"] * store_overlap["weight"]).sum()

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

def penalize_new_store_clustering(candidate, candidate_cells):
    penalty = 0
    for _, other in candidate_cells.iterrows():
        if candidate["cell_idx"] == other["cell_idx"]:
            continue
        dist = candidate.geometry.centroid.distance(other.geometry.centroid)
        if dist < CELLSIZE:
            penalty += 1
    return penalty

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
        exisitng_store_penality = np.exp(-5 * coverage_ratio)  

        health = (overlap["Estimate"] * overlap["weight"]).sum()

        dists = candidate_cells.geometry.centroid.distance(candidate.geometry.centroid)
        cluster_penalty = np.sum(np.exp(-dists / CELLSIZE)) - 1
        cluster_factor = np.exp(-cluster_penalty)
        population_term = np.sqrt(population)


        #print("Weights:", overlap["weight"].values) 
        #print("Sum of weights:", overlap["weight"].sum())

        # so this would be (0.5 * 10000)(1 - 0.7) = 1,500
        #  So placing a store here only gives you 30% of the reward you'd get in a completely uncovered area.
        #score = ((0.5 * population) + exisitng_store_penality + income_fitness_func(candidate) * population) * (1 + health/100)  # add 1 to make sure we don't lose points for negative health scores
        score = (population_term * (income_fitness_func(candidate) * population)) \
        * exisitng_store_penality  * cluster_factor \
        * (1 + health/100)
        #print(f"Candidate cell {candidate['cell_idx']}")
        #print(f"Population {population:.2f})")
        #print(f"Health {health:.2f}")
        #print(f"Neighborhood: {overlap['BlockGr202'].tolist()}")
       
        total_score += score
    return total_score

## Functions for stats/reporting

# given any geometry, tells you how many people live inside it, how many are low income, and the averages.
def get_stats_for_area_coverage(coverage_shape):
    if coverage_shape.is_empty:
        return {"population": 0, "low_income_pop": 0, "avg_income": 0, "avg_hyp": 0}
    shape_gdf = gpd.GeoDataFrame(geometry=[coverage_shape], crs=neighborhoods.crs)
    overlap = gpd.overlay(neighborhoods, shape_gdf, how="intersection")

    overlap["overlap_area"] = overlap.geometry.area
    overlap["weight"] = overlap["overlap_area"] / overlap["orig_area"]

    # weighted population = neighborhood pop. * fraction of that neighborhood inside the shape
    weighted_pop = overlap["Total Population"] * overlap["weight"]
    total_pop = weighted_pop.sum()

    # how many of those people are in low-income neighborhoods
    low_income_pop = (overlap["Total Population"] * overlap["weight"] * (overlap["Median Household Income"] < 95000)).sum()

    if total_pop > 0:
        # average income and health of the population covered by the shape, weighted by how much of each neighborhood is covered
        avg_income = (overlap["Median Household Income"] * weighted_pop).sum() / total_pop
        avg_hyp = (overlap["Estimate"] * weighted_pop).sum() / total_pop
    else:
        avg_income = 0
        avg_hyp = 0

    return {
        "population": total_pop,
        "low_income_pop": low_income_pop,
        "avg_income": avg_income,
        "avg_hyp": avg_hyp
    }

def get_original_boston_stats():

    total_bos_pop = neighborhoods["Total Population"].sum()

    # draw a circle around each existing store and merge it
    store_areas = [g.buffer(CELLSIZE) for g in stores_gdf.geometry]
    covered_area = gpd.GeoDataFrame(geometry=store_areas, crs=stores_gdf.crs).union_all()

    # count people in merged area
    existing = get_stats_for_area_coverage(covered_area)

    # city-wide averages (weighted by population)
    avg_income = (neighborhoods["Median Household Income"] * neighborhoods["Total Population"]).sum() / total_bos_pop
    avg_hyp = (neighborhoods["Estimate"] * neighborhoods["Total Population"]).sum() / total_bos_pop

    return {
        "total_pop": total_bos_pop,
        "already_served": existing["population"],
        "covered_area": covered_area,
        "avg_income": avg_income,
        "avg_hyp": avg_hyp,
    }

# evaluate a chromose. What do our 10 new stores actually add. pass in the original
# boston stat dict so we don't have to recalculate the original coverage every time.
def evaluate_chromosome(chromosome, original_stats):
    store_cells = candidates[chromosome == 1].copy()
    city_avg_hyp = neighborhoods["Estimate"].mean()

    # draw a buffer circle around each of our 10 stores, merge into one shape
    new_circles = [cell.geometry.buffer(CELLSIZE) for _, cell in store_cells.iterrows()]
    our_area = gpd.GeoSeries(new_circles, crs=candidates.crs).union_all()

    # new area will subtract the area already covered by existing stores from the area covered by our new stores.
    new_area = our_area.difference(original_stats["covered_area"])
    new_coverage = get_stats_for_area_coverage(new_area)

    # check each store: is it in a low-income or high-hypertension zone?
    low_income_count = 0
    high_hyp_count = 0
    per_store = []
 
    for _, cell in store_cells.iterrows():
        stats = get_stats_for_area_coverage(cell.geometry.buffer(CELLSIZE))
        is_low_income = stats["avg_income"] < 95000
        is_high_hyp = stats["avg_hyp"] > city_avg_hyp
 
        if is_low_income:
            low_income_count += 1
        if is_high_hyp:
            high_hyp_count += 1
 
        per_store.append({
            "cell": cell["cell_idx"],
            "pop": stats["population"],
            "income": stats["avg_income"],
            "hyp": stats["avg_hyp"],
            "low_income": is_low_income,
            "high_hyp": is_high_hyp,
        })

    return {
        "total_pop": original_stats["total_pop"],
            "already_served": original_stats["already_served"],
            "new_people": new_coverage["population"],
            "low_income_stores": low_income_count,
            "high_hyp_stores": high_hyp_count,
            "n_stores": len(store_cells),
            "new_avg_income": new_coverage["avg_income"],
            "new_avg_hyp": new_coverage["avg_hyp"],
            "new_low_income_pop": new_coverage["low_income_pop"],
            "stores": per_store,
            "city_avg_income": original_stats["avg_income"],
            "city_avg_hyp": original_stats["avg_hyp"],
            }

# finally print function 
def print_results(r):
    percent_served = 100 * r["already_served"] / r["total_pop"]
    percent_new = 100 * r["new_people"] / r["total_pop"]
 
    print(f"boston population: {r['total_pop']:>10,.0f}")
    print(f"already have a store: {r['already_served']:>10,.0f} ({percent_served:.1f}%)")
    print(f"new people we reach: {r['new_people']:>10,.0f} ({percent_new:.1f}%)")
    print(f"people served in total after new placement: {(r['already_served'] + r['new_people']):>10,.0f} ({percent_served + percent_new:.1f}%)")
    print(f"avg income in boston: ${r['city_avg_income']:>9,.0f}")
    print(f"avg hypertension in boston: {r['city_avg_hyp']:.2f}") 
    print()
    print(f"stores in low-income areas: {r['low_income_stores']} / {r['n_stores']}")
    print(f"stores in high-hypertension areas: {r['high_hyp_stores']} / {r['n_stores']}")
    print(f"avg income in new coverage: ${r['new_avg_income']:>9,.0f}")
    print(f"avg hypertension in new coverage: {r['new_avg_hyp']:.2f}")
    print()
    print("per store:")
    for s in r["stores"]:
        tags = []
        if s["low_income"]:
            tags.append("LOW INCOME")
        if s["high_hyp"]:
            tags.append("HIGH HYPERTENSION")
        tag_str = f"  [{', '.join(tags)}]" if tags else ""
        print(f"  cell {s['cell']:>3} | pop {s['pop']:>8,.0f} | income ${s['income']:>9,.0f} | hyp {s['hyp']:.2f}{tag_str}")




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

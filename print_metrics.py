from state_space_representation import neighborhoods, stores_gdf, candidates, CELLSIZE
import geopandas as gpd

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

        # find the neighborhood with the most overlap
        cell_buffer = cell.geometry.buffer(CELLSIZE)
        cell_gdf = gpd.GeoDataFrame(geometry=[cell_buffer], crs=candidates.crs)
        cell_overlap = gpd.overlay(neighborhoods, cell_gdf, how="intersection")
        if not cell_overlap.empty:
            cell_overlap["overlap_area"] = cell_overlap.geometry.area
            top_neighborhood = cell_overlap.loc[cell_overlap["overlap_area"].idxmax(), "BlockGr202"]
        else:
            top_neighborhood = "Unknown"

        per_store.append({
            "cell": cell["cell_idx"],
            "neighborhood": top_neighborhood,
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
        print(f"  cell {s['cell']:>3} | {s['neighborhood']:<15} | pop {s['pop']:>8,.0f} | income ${s['income']:>9,.0f} | hyp {s['hyp']:.2f}{tag_str}")
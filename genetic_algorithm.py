import numpy as np
import matplotlib.pyplot as plt
import warnings
from state_space_representation import candidates, neighborhoods, fitness_func, CELLSIZE, n
from print_metrics import get_original_boston_stats, evaluate_chromosome, print_results

warnings.filterwarnings("ignore", category=RuntimeWarning)

#GA Parameters
N_STORES       = 10     # number of stores to place on the grid
POP_SIZE       = 50     # chromosomes per generation
N_GENERATIONS  = 60     # total generations to run
ELITE_K        = 10     # top-k chromosomes preserved via elitism
TOURNAMENT_K   = 4      # contestants per tournament selection round
MUTATION_RATE  = 0.7    # probability that a child is mutated
MULTISWAP_PROB = 0.15   # probability that a mutation swaps multiple cells
N_MULTISWAP    = 3      # number of swaps in a multi-swap mutation


# Random Chromosome

def random_chromosome():
    """Create a binary chromosome with exactly N_STORES ones at random positions."""
    chrom = np.zeros(n, dtype=int)
    chrom[np.random.choice(n, size=N_STORES, replace=False)] = 1
    return chrom


#Mutation
def mutate_swap(chromosome, n_swaps=1):
    """Swap `n_swaps` store cells with empty cells to explore nearby solutions."""
    mutant = chromosome.copy()
    store_idxs = np.where(mutant == 1)[0]
    empty_idxs = np.where(mutant == 0)[0]
    n_swaps = min(n_swaps, len(store_idxs), len(empty_idxs))

    removes = np.random.choice(store_idxs, size=n_swaps, replace=False)
    adds    = np.random.choice(empty_idxs, size=n_swaps, replace=False)
    mutant[removes] = 0
    mutant[adds]    = 1
    return mutant


def mutate(chromosome, multiswap_prob=MULTISWAP_PROB, n_multiswap=N_MULTISWAP):
    """Apply mutation with a chance of single-swap or multi-swap."""
    if np.random.rand() > MUTATION_RATE:
        return chromosome.copy()
    if np.random.rand() < multiswap_prob:
        return mutate_swap(chromosome, n_swaps=n_multiswap)
    return mutate_swap(chromosome, n_swaps=1)


#Selection

def selection(pop, fitness, k):
    """
    Tournament selection: pick k random chromosomes and return the fittest.
    Less greedy than always picking the global best, which helps avoid local optima.
    """
    competitors = np.random.choice(len(pop), size=k, replace=False)
    winner = competitors[0]
    for i in competitors:
        if fitness[i] > fitness[winner]:
            winner = i
    return pop[winner]


#Main GA loop

def run_ga():
    """Run the genetic algorithm and return the best chromosome and history."""

    # initialise population and evaluate fitness
    population = [random_chromosome() for _ in range(POP_SIZE)]
    fitnesses  = [fitness_func(c) for c in population]

    best_score      = max(fitnesses)
    best_chromosome = population[np.argmax(fitnesses)].copy()
    best_history    = [best_score]
    avg_history     = [np.mean(fitnesses)]

    # adaptive-mutation state (ramps up when progress stalls)
    plateau_counter = 0
    multiswap_prob  = MULTISWAP_PROB
    n_multiswap     = N_MULTISWAP

    # header for progress log
    print(f"{'Gen':>5}  {'Best':>12}  {'Avg':>12}  {'Plateau':>7}")
    print("-" * 45)
    print(f"{0:>5}  {best_score:>12.0f}  {np.mean(fitnesses):>12.0f}  {0:>7}")

    for gen in range(1, N_GENERATIONS + 1):

        # preserve the top ELITE_K chromosomes unchanged
        elite_idxs = np.argsort(fitnesses)[-ELITE_K:]
        new_pop    = [population[i].copy() for i in elite_idxs]

        # fill the rest via tournament selection + mutation
        while len(new_pop) < POP_SIZE:
            child = selection(population, fitnesses, TOURNAMENT_K)
            new_pop.append(child)
            child = mutate(child, multiswap_prob, n_multiswap)
            new_pop.append(child)

        population = new_pop
        fitnesses  = [fitness_func(c) for c in population]

        # track generation stats
        gen_best = max(fitnesses)
        gen_avg  = np.mean(fitnesses)
        best_history.append(gen_best)
        avg_history.append(gen_avg)

        # update global best
        if gen_best > best_score:
            best_score      = gen_best
            best_chromosome = population[np.argmax(fitnesses)].copy()
            plateau_counter = 0
            multiswap_prob  = MULTISWAP_PROB
            n_multiswap     = N_MULTISWAP
        else:
            plateau_counter += 1

        # adaptive mutation: increase exploration intensity when stuck
        if plateau_counter > 20:
            multiswap_prob = min(multiswap_prob + 0.05, 0.5)
            n_multiswap    = min(n_multiswap + 1, N_STORES // 2)

        # log progress every 10 generations and on the final generation
        if gen % 10 == 0 or gen == N_GENERATIONS:
            print(f"{gen:>5}  {gen_best:>12.0f}  {gen_avg:>12.0f}  {plateau_counter:>7}")

    print(f"\nBest fitness:        {best_score:,.0f}")
    print(f"Store cell indices:  {np.where(best_chromosome == 1)[0].tolist()}")
    return best_chromosome, best_score, best_history, avg_history


#Visualisation & evaluation
if __name__ == "__main__":
    best_chrom, best_score, best_hist, avg_hist = run_ga()

    #convergence plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(best_hist, label="Best fitness",    color="#2563eb", linewidth=2)
    ax.plot(avg_hist,  label="Average fitness", color="#94a3b8", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Generation")
    ax.set_ylabel("Fitness score")
    ax.set_title("GA Convergence")
    ax.legend()
    plt.tight_layout()
    plt.savefig("ga_convergence.png", dpi=150)
    plt.show()

    #map of best solution
    candidates["has_store"] = best_chrom
    fig, ax = plt.subplots(figsize=(10, 10))
    neighborhoods.plot(ax=ax, edgecolor="black", linewidth=1.2,
                       color="#dbeafe", alpha=0.6)
    candidates[candidates["has_store"] == 0].plot(
        ax=ax, edgecolor="#cbd5e1", linewidth=0.3, color="#f8fafc", alpha=0.3)
    candidates[candidates["has_store"] == 1].plot(
        ax=ax, edgecolor="#1d4ed8", linewidth=2, color="#3b82f6", alpha=0.9)
    ax.set_title(f"Best store placement  (fitness = {best_score:,.0f})")
    plt.tight_layout()
    plt.savefig("best_placement.png", dpi=150)
    plt.show()

    #print evaluation metrics for the best chromosome
    print(f"\n=== final best | score: {best_score:,.0f} ===")
    og_stats = get_original_boston_stats()
    results  = evaluate_chromosome(best_chrom, og_stats)
    print_results(results)
    print(results["new_people"])
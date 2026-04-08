import numpy as np
import matplotlib.pyplot as plt
from state_space_representation import candidates, neighborhoods, fitness_func, CELLSIZE, n

# ── config ───────────────────────────────────────────────────────────────────
N_STORES       = 10
POP_SIZE       = 50
N_GENERATIONS  = 30
ELITE_K        = 5
TOURNAMENT_K   = 4
MUTATION_RATE  = 0.7
MULTISWAP_PROB = 0.15
N_MULTISWAP    = 3

# ── chromosome helpers ────────────────────────────────────────────────────────
def random_chromosome():
    chrom = np.zeros(n, dtype=int)
    chrom[np.random.choice(n, size=N_STORES, replace=False)] = 1
    return chrom

# ── mutation ──────────────────────────────────────────────────────────────────
def mutate_swap(chromosome, n_swaps=1):
    mutant = chromosome.copy()
    store_idxs = np.where(mutant == 1)[0]
    empty_idxs = np.where(mutant == 0)[0]
    n_swaps = min(n_swaps, len(store_idxs), len(empty_idxs))
    removes = np.random.choice(store_idxs, size=n_swaps, replace=False)
    adds    = np.random.choice(empty_idxs,  size=n_swaps, replace=False)
    mutant[removes] = 0
    mutant[adds]    = 1
    return mutant

def mutate(chromosome, multiswap_prob=0.15, n_multiswap=3):
    if np.random.rand() > MUTATION_RATE:
        return chromosome.copy()
    if np.random.rand() < multiswap_prob:
        return mutate_swap(chromosome, n_swaps=n_multiswap)
    return mutate_swap(chromosome, n_swaps=1)

# ── crossover ─────────────────────────────────────────────────────────────────
#def crossover(parent_a, parent_b):
    child    = np.zeros(n, dtype=int)
    both_on  = np.where((parent_a == 1) & (parent_b == 1))[0]
    a_only   = np.where((parent_a == 1) & (parent_b == 0))[0]
    b_only   = np.where((parent_a == 0) & (parent_b == 1))[0]
    child[both_on] = 1
    still_needed = N_STORES - len(both_on)
    if still_needed > 0:
        contested = np.concatenate([a_only, b_only])
        np.random.shuffle(contested)
        child[contested[:still_needed]] = 1
    return child

# ── selection ─────────────────────────────────────────────────────────────────
def tournament_select(population, fitnesses):
    idxs = np.random.choice(len(population), size=TOURNAMENT_K, replace=False)
    best = idxs[np.argmax([fitnesses[i] for i in idxs])]
    return population[best]

# ── main GA loop ──────────────────────────────────────────────────────────────
def run_ga():
    population = [random_chromosome() for _ in range(POP_SIZE)]
    fitnesses  = [fitness_func(c) for c in population]

    best_score      = max(fitnesses)
    best_chromosome = population[np.argmax(fitnesses)].copy()
    best_history    = [best_score]
    avg_history     = [np.mean(fitnesses)]

    plateau_counter = 0
    multiswap_prob  = MULTISWAP_PROB
    n_multiswap     = N_MULTISWAP

    print(f"{'Gen':>5}  {'Best':>12}  {'Avg':>12}  {'Plateau':>7}")
    print("-" * 45)
    print(f"{0:>5}  {best_score:>12.0f}  {np.mean(fitnesses):>12.0f}  {0:>7}")

    for gen in range(1, N_GENERATIONS + 1):

        # elitism
        elite_idxs = np.argsort(fitnesses)[-ELITE_K:]
        new_pop    = [population[i].copy() for i in elite_idxs]

        # fill rest of population
        while len(new_pop) < POP_SIZE:
            pa    = tournament_select(population, fitnesses)
            pb    = tournament_select(population, fitnesses)
            child = pa.copy()
            child = mutate(child, multiswap_prob, n_multiswap)
            new_pop.append(child)

        population = new_pop
        fitnesses  = [fitness_func(c) for c in population]

        gen_best = max(fitnesses)
        gen_avg  = np.mean(fitnesses)
        best_history.append(gen_best)
        avg_history.append(gen_avg)

        if gen_best > best_score:
            best_score      = gen_best
            best_chromosome = population[np.argmax(fitnesses)].copy()
            plateau_counter = 0
            multiswap_prob  = MULTISWAP_PROB  # reset
            n_multiswap     = N_MULTISWAP
        else:
            plateau_counter += 1

        # adaptive mutation — kick harder if stuck
        if plateau_counter > 20:
            multiswap_prob = min(multiswap_prob + 0.05, 0.5)
            n_multiswap    = min(n_multiswap + 1, N_STORES // 2)

        if gen % 10 == 0 or gen == N_GENERATIONS:
            print(f"{gen:>5}  {gen_best:>12.0f}  {gen_avg:>12.0f}  {plateau_counter:>7}")

    print(f"\nBest fitness:        {best_score:,.0f}")
    print(f"Store cell indices:  {np.where(best_chromosome == 1)[0].tolist()}")
    return best_chromosome, best_score, best_history, avg_history


if __name__ == "__main__":
    best_chrom, best_score, best_hist, avg_hist = run_ga()

    # convergence plot
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

    # map of best solution
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


#!/usr/bin/env python3
"""
Sickle-Cell / HbS Gene Monte-Carlo Simulation
===============================================

Exact Python reproduction of two C programs:
  - GENE2.C   → Paper 1 (Habibzadeh, Sci Rep 2024; 14:5800)
  - GENE2-dis.C → Paper 2 (Habibzadeh, Sci Rep 2024; 14:15399)
Plus the post-processing program fgene2.C

Usage:
    python sickle_cell_sim.py --mode paper1 --repeats 10000 --n_jobs -1 --plot
    python sickle_cell_sim.py --mode paper2 --repeats 10000 --n_jobs -1 --plot

Author: Reproduced from original C code by F. Habibzadeh
"""

import numpy as np
import time
import argparse
import sys
import os
from concurrent.futures import ProcessPoolExecutor
from functools import partial

# =============================================================================
#  ALL SIMULATION PARAMETERS (modify these as needed)
# =============================================================================

# --- Population ---
COUPLES = 25                  # Baseline number of couples
COUPLES_MAX = 1000            # Maximum couples (carrying capacity K)

# --- Simulation control ---
REPEAT = 200                  # Number of Monte-Carlo repetitions
MAX_GEN = 100                 # Number of generations per repetition

# --- Generational overlap ---
OVERLAP = 0.05                # Fraction of overlap between parent and offspring
OVERLAP_START = 1             # Generation from which overlap starts

# --- Population growth ---
GROWTH_START = 5              # Generation from which logistic growth starts
GROWTH_RATE = 0.15            # Logistic growth rate (r)

# --- Lifestyle transition ---
LIFE_STYLE_GEN = 5            # Generation at which hunter → farmer transition occurs
HUNTER = 0
FARMER = 1

# --- Random number precision ---
PREC = 1000                   # Precision of random draws (1/PREC resolution)

# --- Mortality and protection (Paper 1: malaria-only model) ---
M_MALARIA = 0.15              # Mortality rate of malaria in healthy (AA) persons
P_MINOR = 10                  # Protective factor of AS genotype against malaria
P_MAJOR = 10                  # Protective factor of SS genotype against malaria
M_SS = 0.85                   # Mortality from homozygous sickle cell disease

# --- Additional parameters for Paper 2: multi-disease model ---
PR_m = 0.4                    # Prevalence of malaria in the population
# In Paper 2, the conditional mortality given malaria = M_MALARIA / PR_m
# so that the unconditional malaria mortality remains M_MALARIA = 0.15
M_OTHER = 0.25                # Mortality from other (non-malaria) causes
P_m_OTHER = 1.5               # Protective factor of malaria against other diseases
P_MINOR_OTHER = 3.0           # Protective factor of AS against other diseases
P_MAJOR_OTHER = 3.0           # Protective factor of SS against other diseases

# --- Children per family ---
_CHILD = 7                    # Maximum children per family
_MORE_CHILD = 0               # Extra children if HbS increases fertility
CHILD = _CHILD + _MORE_CHILD
MORE_FERTILE = 0.0            # Probability heterozygotes have 1 extra child (Paper 1)
MORE_FERTILE_DIS = 0.0        # Same for Paper 2 (set to 0 in original)

# --- Mutation vs initial frequency ---
USE_MUTATION = True           # True: start with a single mutation; False: use GENE_F0
GENE_F0 = 0.01                # Initial gene frequency if USE_MUTATION is False

# --- Child distribution (Hunter-Gatherer and Farmer) ---
# Each entry: (number_of_children, cumulative_probability_out_of_100)
CHILD_DIST_HUNTER = [(2, 10), (3, 25), (4, 75), (5, 90), (6, 100)]
CHILD_DIST_FARMER = [(3, 10), (4, 25), (5, 75), (6, 90), (7, 100)]

# --- Genotype codes ---
HLTHY = 0
MINOR = 1
MAJOR = 2
DEAD = 4
MALARIA_FLAG = 8
NO_CHILD = -1

# =============================================================================
#  Park-Miller "Minimal Standard" PRNG (replicates the C implementation)
# =============================================================================

class ParkMillerRNG:
    """Park-Miller 31-bit PRNG, identical to rand31_next() in the C code."""

    def __init__(self, seed):
        self.state = int(seed) & 0x7FFFFFFF
        if self.state == 0:
            self.state = 1

    def next(self):
        lo = 16807 * (self.state & 0xFFFF)
        hi = 16807 * (self.state >> 16)
        lo += (hi & 0x7FFF) << 16
        lo += hi >> 15
        if lo > 0x7FFFFFFF:
            lo -= 0x7FFFFFFF
        self.state = lo
        return self.state

    def randint(self, n):
        """Return random integer in [0, n)."""
        return self.next() % n


# =============================================================================
#  Logistic growth function
# =============================================================================

def growth(generation, couples=COUPLES, couples_max=COUPLES_MAX,
           growth_start=GROWTH_START, r=GROWTH_RATE):
    """Logistic population growth: returns total number of individuals (= 2 * couples)."""
    if generation < growth_start:
        generation = growth_start
    N0 = couples
    K = couples_max
    exp_val = np.exp(r * (generation - growth_start))
    coups = K * N0 * exp_val / (K + N0 * (exp_val - 1))
    return int(coups + 0.5) * 2


# =============================================================================
#  Shuffle (replicates the C shuffle: 7 passes of Fisher-Yates-like swap)
# =============================================================================

def shuffle_array(arr, size, rng):
    """Shuffle first `size` elements of arr using 7 full passes."""
    for _ in range(7):
        for i in range(size):
            j = rng.randint(size)
            arr[i], arr[j] = arr[j], arr[i]


# =============================================================================
#  Manage population: partition dead to end, ensure even count
# =============================================================================

def manage_population(parent, N, rng):
    """Remove dead individuals, return new population size (always even)."""
    i, j = 0, N - 1
    while i < j:
        while i < j and (parent[i] & DEAD) != DEAD:
            i += 1
        while j > i and (parent[j] & DEAD) == DEAD:
            j -= 1
        if i < j and (parent[j] & DEAD) != DEAD:
            parent[i], parent[j] = parent[j], parent[i]

    if (parent[i] & DEAD) == DEAD:
        i -= 1
    if i % 2 == 0:  # odd-sized array → make even
        i += 1
        parent[i] = parent[rng.randint(i)]
    return i + 1


def manage_offspring(offspring, childsize):
    """Compact offspring array: move NO_CHILD entries to end. Return count."""
    i, j = 0, childsize - 1
    while i < j:
        while i < j and offspring[i] != NO_CHILD:
            i += 1
        while j > i and offspring[j] == NO_CHILD:
            j -= 1
        if i < j and offspring[j] != NO_CHILD:
            offspring[i], offspring[j] = offspring[j], offspring[i]

    if offspring[i] == NO_CHILD:
        i -= 1
    return i + 1


# =============================================================================
#  Determine number of children for a couple
# =============================================================================

def get_childnum(rng, lifestyle):
    """Return number of children drawn from the child distribution."""
    prob = rng.randint(100)
    dist = CHILD_DIST_FARMER if lifestyle == FARMER else CHILD_DIST_HUNTER
    for num, cum_prob in dist:
        if prob < cum_prob:
            return num
    return dist[-1][0]


# =============================================================================
#  Single repetition: Paper 1 (malaria-only model, replicating GENE2.C)
# =============================================================================

def run_single_paper1(rep_index, seed):
    """Run one repetition of the Paper 1 simulation. Returns (results_array, gene_aborted)."""
    rng = ParkMillerRNG(seed)
    generations_data = np.zeros((MAX_GEN + 1, 4))  # f_gene, f_hetero, f_homo, f_dead

    N = growth(0)
    parent = [HLTHY] * N

    if USE_MUTATION:
        parent[rng.randint(N)] = MINOR
    else:
        n_major = int(GENE_F0 * GENE_F0 * N + 0.5)
        n_minor = int(2 * GENE_F0 * (1 - GENE_F0) * N + 0.5)
        for idx in range(n_major):
            parent[idx] = MAJOR
        for idx in range(n_major, n_major + n_minor):
            parent[idx] = MINOR
        shuffle_array(parent, N, rng)

    gene_aborted = 0
    prevparent = list(parent)
    prevpopsize = N
    childsize = CHILD * COUPLES_MAX

    for generation in range(MAX_GEN + 1):
        # Count genotypes
        n_minor = sum(1 for x in parent[:N] if x == MINOR)
        n_major = sum(1 for x in parent[:N] if x == MAJOR)

        f_gene = (n_minor + 2 * n_major) * 100.0 / (2 * N)
        f_minor = n_minor * 100.0 / N
        f_major = n_major * 100.0 / N

        # --- Selection ---
        n_dead_hlthy = n_dead_minor = n_dead_major = 0
        ii = 0
        for i in range(N):
            if parent[i] == HLTHY and rng.randint(PREC) < M_MALARIA * PREC:
                parent[i] |= DEAD
                n_dead_hlthy += 1

            if parent[i] == MINOR:
                if rng.randint(PREC) < (M_MALARIA / P_MINOR) * PREC:
                    parent[i] |= DEAD
                    n_dead_minor += 1

            if parent[i] == MAJOR:
                if rng.randint(PREC) < M_SS * PREC or rng.randint(PREC) < (M_MALARIA / P_MAJOR) * PREC:
                    parent[i] |= DEAD
                    n_dead_major += 1

            # Overlap
            if (gene_aborted == 0 and generation >= OVERLAP_START
                    and rng.randint(PREC) < OVERLAP * PREC and ii < prevpopsize):
                parent[i] = prevparent[ii]
                ii += 1

        n_dead = sum(1 for x in parent[:N] if (x & DEAD) == DEAD)
        f_dead = n_dead * 100.0 / N

        generations_data[generation] = [f_gene, f_minor, f_major, f_dead]

        # Check abort
        n_minor_alive = sum(1 for x in parent[:N] if x == MINOR)
        n_major_alive = sum(1 for x in parent[:N] if x == MAJOR)
        if gene_aborted == 0 and n_minor_alive == 0 and n_major_alive == 0:
            gene_aborted = generation

        # --- Mating ---
        popsize = N - n_dead
        if popsize <= 1:
            if gene_aborted == 0:
                gene_aborted = generation
            if N - n_dead > 0:
                n_minor_surv = n_minor - n_dead_minor
                n_major_surv = n_major - n_dead_major
                n_minor_new = int(n_minor_surv * N / (N - n_dead) + 0.5)
                n_major_new = int(n_major_surv * N / (N - n_dead) + 0.5)
            else:
                n_minor_new = 0
                n_major_new = 0
            parent = [HLTHY] * N
            for idx in range(min(n_major_new, N)):
                parent[idx] = MAJOR
            for idx in range(n_major_new, min(n_major_new + n_minor_new, N)):
                parent[idx] = MINOR
            shuffle_array(parent, N, rng)
            popsize = N
        else:
            popsize = manage_population(parent, N, rng)

        shuffle_array(parent, popsize, rng)

        offspring = [NO_CHILD] * childsize
        lifestyle = FARMER if generation >= LIFE_STYLE_GEN else HUNTER

        for i in range(0, popsize - 1, 2):
            j = i + 1
            childnum = get_childnum(rng, lifestyle)

            more_child = _MORE_CHILD
            if (parent[i] == MINOR or parent[j] == MINOR) and rng.randint(PREC) < MORE_FERTILE * PREC:
                more_child = 0

            chbase = i * CHILD // 2

            pi = parent[i]
            pj = parent[j]

            if pi == HLTHY:
                if pj == HLTHY:
                    for k in range(_MORE_CHILD, childnum + _MORE_CHILD):
                        offspring[chbase + k] = HLTHY
                elif pj == MINOR:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        offspring[chbase + k] = rng.randint(2)
                elif pj == MAJOR:
                    for k in range(_MORE_CHILD, childnum + _MORE_CHILD):
                        offspring[chbase + k] = MINOR
            elif pi == MINOR:
                if pj == HLTHY:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        offspring[chbase + k] = rng.randint(2)
                elif pj == MINOR:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        r_val = rng.randint(4)
                        if r_val == 0:
                            offspring[chbase + k] = HLTHY
                        elif r_val <= 2:
                            offspring[chbase + k] = MINOR
                        else:
                            offspring[chbase + k] = MAJOR
                if pj == MAJOR:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        offspring[chbase + k] = 1 + rng.randint(2)
            elif pi == MAJOR:
                if pj == HLTHY:
                    for k in range(_MORE_CHILD, childnum + _MORE_CHILD):
                        offspring[chbase + k] = MINOR
                elif pj == MINOR:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        offspring[chbase + k] = 1 + rng.randint(2)
                elif pj == MAJOR:
                    for k in range(_MORE_CHILD, childnum + _MORE_CHILD):
                        offspring[chbase + k] = MAJOR

        prevparent = list(parent[:popsize])
        prevpopsize = popsize
        shuffle_array(prevparent, prevpopsize, rng)

        total_offspring = manage_offspring(offspring, childsize)
        shuffle_array(offspring, total_offspring, rng)
        shuffle_array(offspring, total_offspring, rng)

        N = growth(generation + 1)
        parent = [HLTHY] * max(N, len(parent))
        for i in range(N):
            if i < total_offspring and offspring[i] != NO_CHILD:
                parent[i] = offspring[i]
            else:
                parent[i] = offspring[rng.randint(total_offspring)]

    return generations_data, gene_aborted


# =============================================================================
#  Single repetition: Paper 2 (multi-disease model, replicating GENE2-dis.C)
# =============================================================================

def run_single_paper2(rep_index, seed):
    """Run one repetition of the Paper 2 simulation. Returns (results_array, gene_aborted)."""
    rng = ParkMillerRNG(seed)
    M_cond = M_MALARIA / PR_m  # conditional mortality given malaria

    generations_data = np.zeros((MAX_GEN + 1, 4))

    N = growth(0)
    parent = [HLTHY] * N

    if USE_MUTATION:
        parent[rng.randint(N)] = MINOR
    else:
        n_major = int(GENE_F0 * GENE_F0 * N + 0.5)
        n_minor = int(2 * GENE_F0 * (1 - GENE_F0) * N + 0.5)
        for idx in range(n_major):
            parent[idx] = MAJOR
        for idx in range(n_major, n_major + n_minor):
            parent[idx] = MINOR
        shuffle_array(parent, N, rng)

    gene_aborted = 0
    prevparent = list(parent)
    prevpopsize = N
    childsize = CHILD * COUPLES_MAX

    for generation in range(MAX_GEN + 1):
        # Count genotypes (using mask)
        n_minor = sum(1 for x in parent[:N] if (x & (MINOR | MAJOR)) == MINOR)
        n_major = sum(1 for x in parent[:N] if (x & (MINOR | MAJOR)) == MAJOR)

        f_gene = (n_minor + 2 * n_major) * 100.0 / (2 * N)
        f_minor = n_minor * 100.0 / N
        f_major = n_major * 100.0 / N

        # --- Assign malaria status ---
        for i in range(N):
            if rng.randint(PREC) < PR_m * PREC:
                parent[i] |= MALARIA_FLAG

        # --- Selection ---
        n_dead_hlthy = n_dead_minor = n_dead_major = 0
        ii = 0
        for i in range(N):
            genotype = parent[i] & (MINOR | MAJOR)
            has_malaria = (parent[i] & MALARIA_FLAG) != 0

            if has_malaria:
                if genotype == HLTHY:
                    if rng.randint(PREC) < M_cond * PREC or rng.randint(PREC) < (M_OTHER / P_m_OTHER) * PREC:
                        parent[i] |= DEAD
                        n_dead_hlthy += 1
                elif genotype == MINOR:
                    if rng.randint(PREC) < (M_cond / P_MINOR) * PREC or rng.randint(PREC) < (M_OTHER / (P_MINOR_OTHER * P_m_OTHER)) * PREC:
                        parent[i] |= DEAD
                        n_dead_minor += 1
                elif genotype == MAJOR:
                    if rng.randint(PREC) < (M_cond / P_MAJOR) * PREC or rng.randint(PREC) < M_SS * PREC or rng.randint(PREC) < (M_OTHER / (P_MAJOR_OTHER * P_m_OTHER)) * PREC:
                        parent[i] |= DEAD
                        n_dead_major += 1
            else:  # no malaria
                if genotype == HLTHY:
                    if rng.randint(PREC) < M_OTHER * PREC:
                        parent[i] |= DEAD
                        n_dead_hlthy += 1
                elif genotype == MINOR:
                    if rng.randint(PREC) < (M_OTHER / P_MINOR_OTHER) * PREC:
                        parent[i] |= DEAD
                        n_dead_minor += 1
                elif genotype == MAJOR:
                    if rng.randint(PREC) < M_SS * PREC or rng.randint(PREC) < (M_OTHER / P_MAJOR_OTHER) * PREC:
                        parent[i] |= DEAD
                        n_dead_major += 1

            # Overlap
            if (gene_aborted == 0 and generation >= OVERLAP_START
                    and rng.randint(PREC) < OVERLAP * PREC and ii < prevpopsize):
                parent[i] = prevparent[ii]
                ii += 1

        n_dead = sum(1 for x in parent[:N] if (x & DEAD) == DEAD)
        f_dead = n_dead * 100.0 / N

        generations_data[generation] = [f_gene, f_minor, f_major, f_dead]

        # Check abort (matches C code: uses plain switch, no mask)
        n_minor_alive = sum(1 for x in parent[:N] if (x & ~MALARIA_FLAG) == MINOR)
        n_major_alive = sum(1 for x in parent[:N] if (x & ~MALARIA_FLAG) == MAJOR)
        if gene_aborted == 0 and n_minor_alive == 0 and n_major_alive == 0:
            gene_aborted = generation

        # --- Mating ---
        popsize = N - n_dead
        if popsize <= 1:
            if gene_aborted == 0:
                gene_aborted = generation
            if N - n_dead > 0:
                n_minor_surv = n_minor - n_dead_minor
                n_major_surv = n_major - n_dead_major
                n_minor_new = int(n_minor_surv * N / (N - n_dead) + 0.5)
                n_major_new = int(n_major_surv * N / (N - n_dead) + 0.5)
            else:
                n_minor_new = 0
                n_major_new = 0
            parent = [HLTHY] * N
            for idx in range(min(n_major_new, N)):
                parent[idx] = MAJOR
            for idx in range(n_major_new, min(n_major_new + n_minor_new, N)):
                parent[idx] = MINOR
            shuffle_array(parent, N, rng)
            popsize = N
        else:
            popsize = manage_population(parent, N, rng)

        shuffle_array(parent, popsize, rng)

        offspring = [NO_CHILD] * childsize
        lifestyle = FARMER if generation >= LIFE_STYLE_GEN else HUNTER

        # Use MORE_FERTILE_DIS for Paper 2
        more_fertile_prob = MORE_FERTILE_DIS

        for i in range(0, popsize - 1, 2):
            j_idx = i + 1
            childnum = get_childnum(rng, lifestyle)

            more_child = _MORE_CHILD
            pi_geno = parent[i] & (MINOR | MAJOR)
            pj_geno = parent[j_idx] & (MINOR | MAJOR)

            if (pi_geno == MINOR or pj_geno == MINOR) and rng.randint(PREC) < more_fertile_prob * PREC:
                more_child = 0

            chbase = i * CHILD // 2

            if pi_geno == HLTHY:
                if pj_geno == HLTHY:
                    for k in range(_MORE_CHILD, childnum + _MORE_CHILD):
                        offspring[chbase + k] = HLTHY
                elif pj_geno == MINOR:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        offspring[chbase + k] = rng.randint(2)
                elif pj_geno == MAJOR:
                    for k in range(_MORE_CHILD, childnum + _MORE_CHILD):
                        offspring[chbase + k] = MINOR
            elif pi_geno == MINOR:
                if pj_geno == HLTHY:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        offspring[chbase + k] = rng.randint(2)
                elif pj_geno == MINOR:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        r_val = rng.randint(4)
                        if r_val == 0:
                            offspring[chbase + k] = HLTHY
                        elif r_val <= 2:
                            offspring[chbase + k] = MINOR
                        else:
                            offspring[chbase + k] = MAJOR
                if pj_geno == MAJOR:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        offspring[chbase + k] = 1 + rng.randint(2)
            elif pi_geno == MAJOR:
                if pj_geno == HLTHY:
                    for k in range(_MORE_CHILD, childnum + _MORE_CHILD):
                        offspring[chbase + k] = MINOR
                elif pj_geno == MINOR:
                    for k in range(more_child, childnum + _MORE_CHILD):
                        offspring[chbase + k] = 1 + rng.randint(2)
                elif pj_geno == MAJOR:
                    for k in range(_MORE_CHILD, childnum + _MORE_CHILD):
                        offspring[chbase + k] = MAJOR

        prevparent = list(parent[:popsize])
        prevpopsize = popsize
        shuffle_array(prevparent, prevpopsize, rng)

        total_offspring = manage_offspring(offspring, childsize)
        shuffle_array(offspring, total_offspring, rng)
        shuffle_array(offspring, total_offspring, rng)

        N = growth(generation + 1)
        parent = [HLTHY] * max(N, len(parent))
        for i in range(N):
            if i < total_offspring and offspring[i] != NO_CHILD:
                parent[i] = offspring[i]
            else:
                parent[i] = offspring[rng.randint(total_offspring)]

    return generations_data, gene_aborted


# =============================================================================
#  Post-processing (replicating fgene2.C)
# =============================================================================

def compute_statistics(all_results, all_aborted, repeat):
    """Compute mean, SD for each generation across all non-aborted repetitions."""
    n_gen = MAX_GEN + 1
    n_vars = 4  # f_gene, f_hetero, f_homo, f_dead

    f_abort = sum(1 for a in all_aborted if a != 0)
    n = repeat - f_abort

    if n == 0:
        print("All repetitions aborted!")
        return None

    # Accumulate sums and sums of squares
    Sx = np.zeros((n_gen, n_vars))
    S2x = np.zeros((n_gen, n_vars))

    for res, aborted in zip(all_results, all_aborted):
        if aborted == 0:
            Sx += res
            S2x += res ** 2

    ERROR = 1e-12
    means = Sx / n
    variances = (S2x - Sx ** 2 / n) / (n - 1) + ERROR
    # Clamp negative values (numerical noise)
    variances = np.maximum(variances, 0)
    sds = np.sqrt(variances)

    p_abort = f_abort / repeat
    sem_abort = np.sqrt(p_abort * (1 - p_abort) / repeat)

    stats = {
        'means': means,
        'sds': sds,
        'f_abort': f_abort,
        'n_valid': n,
        'p_abort': p_abort,
        'sem_abort': sem_abort,
    }
    return stats


# =============================================================================
#  Worker for parallel execution
# =============================================================================

def _worker_paper1(args):
    idx, seed = args
    return run_single_paper1(idx, seed)


def _worker_paper2(args):
    idx, seed = args
    return run_single_paper2(idx, seed)


# =============================================================================
#  Main simulation runner with parallel processing
# =============================================================================

def run_simulation(mode='paper1', repeat=REPEAT, n_jobs=-1, seed_base=None, verbose=True):
    """
    Run the full Monte-Carlo simulation.

    Parameters
    ----------
    mode : str
        'paper1' for GENE2.C model, 'paper2' for GENE2-dis.C model
    repeat : int
        Number of repetitions
    n_jobs : int
        Number of parallel workers (-1 = all CPUs)
    seed_base : int or None
        Base seed for reproducibility. If None, use the C program's default.
    verbose : bool
        Print progress

    Returns
    -------
    stats : dict with 'means', 'sds', 'f_abort', 'n_valid', 'p_abort', 'sem_abort'
    """
    if seed_base is None:
        if mode == 'paper1':
            seed_base = (repeat + 7 * COUPLES + 3 * MAX_GEN + 11 * _CHILD
                         + 13 * _MORE_CHILD + 17 * OVERLAP_START
                         + 19 * GROWTH_START + 23 * LIFE_STYLE_GEN + 7)
        else:
            seed_base = 123

    if n_jobs == -1:
        n_jobs = os.cpu_count() or 4
    if n_jobs == 1:
        n_jobs = None  # serial mode

    # Generate unique seeds per repetition
    master_rng = np.random.RandomState(seed_base)
    seeds = master_rng.randint(1, 2**31 - 1, size=repeat)

    worker = _worker_paper1 if mode == 'paper1' else _worker_paper2
    tasks = list(zip(range(repeat), seeds))

    if verbose:
        print(f"Running {repeat} repetitions ({mode}) with {n_jobs or 1} workers...")

    t0 = time.time()

    all_results = []
    all_aborted = []

    if n_jobs is None:
        # Serial
        for i, task in enumerate(tasks):
            res, aborted = worker(task)
            all_results.append(res)
            all_aborted.append(aborted)
            if verbose and (i + 1) % 100 == 0:
                print(f"  {i+1}/{repeat}", flush=True)
    else:
        # Parallel
        with ProcessPoolExecutor(max_workers=n_jobs) as executor:
            for i, (res, aborted) in enumerate(executor.map(worker, tasks, chunksize=max(1, repeat // (n_jobs * 4)))):
                all_results.append(res)
                all_aborted.append(aborted)
                if verbose and (i + 1) % 500 == 0:
                    elapsed = time.time() - t0
                    eta = elapsed / (i + 1) * (repeat - i - 1)
                    print(f"  {i+1}/{repeat}  elapsed={elapsed:.1f}s  ETA={eta:.1f}s", flush=True)

    elapsed = time.time() - t0
    if verbose:
        print(f"Completed in {elapsed:.1f}s")

    stats = compute_statistics(all_results, all_aborted, repeat)
    return stats


# =============================================================================
#  CSV output (replicating fgene2.C output format)
# =============================================================================

def save_csv(stats, filename, mode='paper1', repeat=REPEAT):
    """Save results to CSV in the same format as fgene2.C output."""
    with open(filename, 'w') as f:
        p = stats['p_abort']
        sem = stats['sem_abort']
        f.write(f"Mode: {mode}, Repeats: {repeat}\n")
        f.write(f"Frequency of Gene Aborted: ({stats['f_abort']} of {repeat}) "
                f"{p*100:.1f}% (95% CI: {(p-sem)*100:.1f}, {(p+sem)*100:.1f})\n")
        f.write("\nGen, f_Gen, SDf_Gen, f_het, SDf_het, f_hom, SDf_hom, d, SDd\n")
        for gen in range(MAX_GEN + 1):
            m = stats['means'][gen]
            s = stats['sds'][gen]
            f.write(f"{gen:4d}, {m[0]:6.3f}, {s[0]:6.3f}, {m[1]:6.3f}, {s[1]:6.3f}, "
                    f"{m[2]:6.3f}, {s[2]:6.3f}, {m[3]:6.3f}, {s[3]:6.3f}\n")


# =============================================================================
#  Plotting
# =============================================================================

def plot_results(stats, mode='paper1', save_path=None):
    """Plot gene frequency, heterozygote/homozygote frequencies, and mortality."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping plot.")
        return

    gens = np.arange(MAX_GEN + 1)
    means = stats['means']
    sds = stats['sds']

    # Equilibrium gene frequency (theoretical)
    W_AA = 1 - M_MALARIA
    W_AS = 1 - M_MALARIA / P_MINOR
    W_SS = 1 - (M_SS + (1 - M_SS) * M_MALARIA / P_MAJOR)
    p_eq = (W_AA - W_AS) / (W_AA - 2 * W_AS + W_SS) * 100

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Sickle Cell Simulation — {'Paper 1 (malaria only)' if mode == 'paper1' else 'Paper 2 (multi-disease)'}",
                 fontsize=14, fontweight='bold')

    titles = ['Gene Frequency (%)', 'Heterozygote Frequency (%)',
              'Homozygote Frequency (%)', 'Mortality (%)']
    colors = ['#2ca02c', '#1f77b4', '#d62728', '#7f7f7f']

    for ax_idx, ax in enumerate(axes.flat):
        m = means[:, ax_idx]
        s = sds[:, ax_idx]
        ax.plot(gens, m, color=colors[ax_idx], linewidth=2)
        ax.fill_between(gens, m - 1.96 * s, m + 1.96 * s, color=colors[ax_idx], alpha=0.15)
        ax.set_xlabel('Generation')
        ax.set_ylabel(titles[ax_idx])
        ax.set_title(titles[ax_idx])
        ax.grid(True, alpha=0.3)
        if ax_idx == 0:
            ax.axhline(y=p_eq, color='gray', linestyle='--', alpha=0.7, label=f'Equilibrium = {p_eq:.1f}%')
            ax.legend()

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


# =============================================================================
#  CLI entry point
# =============================================================================

def main():
    parser = argparse.ArgumentParser(description='Sickle-Cell HbS Gene Monte-Carlo Simulation')
    parser.add_argument('--mode', choices=['paper1', 'paper2'], default='paper1',
                        help='Simulation mode: paper1 (GENE2.C) or paper2 (GENE2-dis.C)')
    parser.add_argument('--repeats', type=int, default=REPEAT, help='Number of MC repetitions')
    parser.add_argument('--n_jobs', type=int, default=-1, help='Parallel workers (-1=all CPUs, 1=serial)')
    parser.add_argument('--seed', type=int, default=None, help='Base seed for reproducibility')
    parser.add_argument('--csv', type=str, default=None, help='Output CSV file path')
    parser.add_argument('--plot', action='store_true', help='Show/save plot')
    parser.add_argument('--plot_file', type=str, default=None, help='Save plot to file')
    args = parser.parse_args()

    stats = run_simulation(mode=args.mode, repeat=args.repeats,
                           n_jobs=args.n_jobs, seed_base=args.seed)

    if stats is None:
        sys.exit(1)

    csv_file = args.csv or f"fgene_{args.mode}.csv"
    save_csv(stats, csv_file, mode=args.mode, repeat=args.repeats)
    print(f"CSV saved to {csv_file}")

    if args.plot or args.plot_file:
        plot_results(stats, mode=args.mode, save_path=args.plot_file)


if __name__ == '__main__':
    main()

"""
Neolithic HbS / Sickle-Cell Gene Stochastic Simulation
Exact reproduction of both 2024 Scientific Reports papers
Parallel Monte-Carlo version (joblib) - February 2026
"""

import numpy as np
import pandas as pd
import argparse
from tqdm import tqdm
import matplotlib.pyplot as plt
from pathlib import Path
import os

# Parallel processing (fallback if joblib not installed)
try:
    from joblib import Parallel, delayed
    HAS_JOBLIB = True
except ImportError:
    HAS_JOBLIB = False

# Genotype constants (identical to your C code)
HLTHY = 0
MINOR = 1
MAJOR = 2
DEAD = 4
NO_CHILD = -1

CHILDPROB_HUNTER = np.array([[2,10],[3,25],[4,75],[5,90],[6,100]], dtype=int)
CHILDPROB_FARMER = np.array([[3,10],[4,25],[5,75],[6,90],[7,100]], dtype=int)

def growth(generation, GROWTH_START=5, N0=25, K=1000, r=0.15):
    """Logistic population growth (exactly as in your C code)"""
    if generation < GROWTH_START:
        generation = GROWTH_START
    exp_term = np.exp(r * (generation - GROWTH_START))
    coups = K * N0 * exp_term / (K + N0 * (exp_term - 1))
    return int(coups + 0.5) * 2

class SickleCellSimulator:
    def __init__(self, args):
        self.args = args
        self.is_paper2 = (args.mode == 'paper2')
        
        # Parameters (exactly as in your two papers)
        self.PR_m = 0.4 if self.is_paper2 else 0.0
        self.M = 0.15 / self.PR_m if self.is_paper2 else 0.15
        self.M_O = 0.25 if self.is_paper2 else 0.0
        self.P_m_O = 1.5 if self.is_paper2 else 1.0
        self.P_MINOR_O = 3.0 if self.is_paper2 else 1.0
        self.P_MAJOR_O = 3.0 if self.is_paper2 else 1.0
        self.MORE_FERTILE = 0.0 if self.is_paper2 else 0.10
        self.P_MINOR = 10
        self.P_MAJOR = 10
        self.M_SS = 0.85
        self._CHILD = 7
        self.CHILD = self._CHILD
        self.CHILDSIZE = self.CHILD * args.couples_max
        self.PREC = 1000
        self.OVERLAP = 0.05
        self.OVERLAP_START = 1
        self.GROWTH_START = 5
        self.LIFE_STYLE = 5
        self.MAX_GEN = args.max_gen
        self.REPEAT = args.repeats
        self.COUPLES = args.couples
        self.n_jobs = args.n_jobs if hasattr(args, 'n_jobs') else -1

    def _one_full_run(self, rep_seed):
        """Single Monte-Carlo run - fully independent (for parallel)"""
        rng = np.random.default_rng(rep_seed)
        N = growth(0, self.GROWTH_START, self.COUPLES, self.args.couples_max)
        parent = np.full(N, HLTHY, dtype=np.int8)
        parent[rng.integers(N)] = MINOR   # one mutant heterozygote

        prevparent = np.zeros(N, dtype=np.int8)
        prevpopsize = 0
        gene_aborted = 0
        traj = np.zeros((self.MAX_GEN + 1, 4), dtype=float)  # f_gene, f_minor, f_major, f_dead

        for gen in range(self.MAX_GEN + 1):
            # Count carriers
            n_minor = np.sum(parent == MINOR)
            n_major = np.sum(parent == MAJOR)
            n_total = len(parent)
            f_gene = (n_minor + 2 * n_major) * 100.0 / (2 * n_total)
            f_minor = n_minor * 100.0 / n_total
            f_major = n_major * 100.0 / n_total

            # === MORTALITY (exact match to your two C files) ===
            if not self.is_paper2:
                # Paper 1 - simple malaria-only
                for i in range(n_total):
                    if parent[i] == HLTHY and rng.random() < self.M:
                        parent[i] |= DEAD
                    elif parent[i] == MINOR and rng.random() < self.M / self.P_MINOR:
                        parent[i] |= DEAD
                    elif parent[i] == MAJOR and (rng.random() < self.M_SS or rng.random() < self.M / self.P_MAJOR):
                        parent[i] |= DEAD
            else:
                # Paper 2 - malaria + other diseases
                has_mal = rng.random(n_total) < self.PR_m
                for i in range(n_total):
                    p = parent[i]
                    if p & DEAD:
                        continue
                    if has_mal[i]:
                        if p == HLTHY:
                            if rng.random() < self.M or rng.random() < self.M_O / self.P_m_O:
                                parent[i] |= DEAD
                        elif p == MINOR:
                            if rng.random() < self.M / self.P_MINOR or rng.random() < self.M_O / (self.P_MINOR_O * self.P_m_O):
                                parent[i] |= DEAD
                        elif p == MAJOR:
                            if rng.random() < self.M / self.P_MAJOR or rng.random() < self.M_SS or rng.random() < self.M_O / (self.P_MAJOR_O * self.P_m_O):
                                parent[i] |= DEAD
                    else:
                        if p == HLTHY and rng.random() < self.M_O:
                            parent[i] |= DEAD
                        elif p == MINOR and rng.random() < self.M_O / self.P_MINOR_O:
                            parent[i] |= DEAD
                        elif p == MAJOR and (rng.random() < self.M_SS or rng.random() < self.M_O / self.P_MAJOR_O):
                            parent[i] |= DEAD

            # Overlap (5 % generational)
            if gen >= self.OVERLAP_START and gene_aborted == 0:
                n_overlap = int(self.OVERLAP * n_total)
                if n_overlap > 0:
                    overlap_idx = rng.choice(n_total, n_overlap, replace=False)
                    prev_idx = np.arange(min(n_overlap, prevpopsize))
                    parent[overlap_idx[prev_idx]] = prevparent[prev_idx]

            n_dead = np.sum((parent & DEAD) != 0)
            f_dead = n_dead * 100.0 / n_total
            traj[gen] = [f_gene, f_minor, f_major, f_dead]

            if gene_aborted == 0 and n_minor == 0 and n_major == 0:
                gene_aborted = gen

            # Manage population for mating
            live = parent[(parent & DEAD) == 0]
            popsize = len(live)
            if popsize <= 1:
                if gene_aborted == 0:
                    gene_aborted = gen
                parent = np.full(N, HLTHY, dtype=np.int8)
                parent[:n_minor] = MINOR
                parent[n_minor:n_minor+n_major] = MAJOR
                rng.shuffle(parent)
                popsize = N
            else:
                parent = live.copy()
                rng.shuffle(parent)

            prevparent = parent.copy()
            prevpopsize = popsize

            # === REPRODUCTION (exact Mendelian logic from your C code) ===
            offspring = np.full(self.CHILDSIZE, NO_CHILD, dtype=np.int8)
            lifestyle = 1 if gen >= self.LIFE_STYLE else 0
            childprob = CHILDPROB_FARMER if lifestyle else CHILDPROB_HUNTER

            i = 0
            while i < popsize - 1:
                prob = rng.integers(100)
                childnum = childprob[np.searchsorted(childprob[:,1], prob), 0]

                p1 = parent[i]
                p2 = parent[i+1]
                base = i * self.CHILD // 2
                more = 1 if (p1 == MINOR or p2 == MINOR) and rng.random() < self.MORE_FERTILE else 0

                # Mendelian inheritance (identical to your C switch statements)

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

# Genotype constants (identical to original)
HLTHY = 0
MINOR = 1
MAJOR = 2
DEAD = 4
NO_CHILD = -1

CHILDPROB_HUNTER = np.array([[2,10],[3,25],[4,75],[5,90],[6,100]], dtype=int)
CHILDPROB_FARMER = np.array([[3,10],[4,25],[5,75],[6,90],[7,100]], dtype=int)

def growth(generation, GROWTH_START=5, N0=25, K=1000, r=0.15):
    """Logistic population growth"""
    if generation < GROWTH_START:
        generation = GROWTH_START
    exp_term = np.exp(r * (generation - GROWTH_START))
    coups = K * N0 * exp_term / (K + N0 * (exp_term - 1))
    return int(coups + 0.5) * 2

class SickleCellSimulator:
    def __init__(self, args):
        self.args = args
        self.is_paper2 = (args.mode == 'paper2')
        
        # Parameters
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
        """Single Monte-Carlo run - independent for parallel"""
        rng = np.random.default_rng(rep_seed)
        N = growth(0, self.GROWTH_START, self.COUPLES, self.args.couples_max)
        parent = np.full(N, HLTHY, dtype=np.int8)
        parent[rng.integers(N)] = MINOR

        prevparent = np.zeros(N, dtype=np.int8)
        prevpopsize = 0
        gene_aborted = 0
        traj = np.zeros((self.MAX_GEN + 1, 4), dtype=float)  # f_gene, f_minor, f_major, f_dead

        for gen in range(self.MAX_GEN + 1):
            n_minor = np.sum(parent == MINOR)
            n_major = np.sum(parent == MAJOR)
            n_total = len(parent)
            f_gene = (n_minor + 2 * n_major) * 100.0 / (2 * n_total)
            f_minor = n_minor * 100.0 / n_total
            f_major = n_major * 100.0 / n_total

            if not self.is_paper2:
                for i in range(n_total):
                    if parent[i] == HLTHY and rng.random() < self.M:
                        parent[i] |= DEAD
                    elif parent[i] == MINOR and rng.random() < self.M / self.P_MINOR:
                        parent[i] |= DEAD
                    elif parent[i] == MAJOR and (rng.random() < self.M_SS or rng.random() < self.M / self.P_MAJOR):
                        parent[i] |= DEAD
            else:
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
                childnum += more

                if p1 == HLTHY:
                    if p2 == HLTHY:
                        offspring[base:base+childnum] = HLTHY
                    elif p2 == MINOR:
                        offspring[base:base+childnum] = rng.integers(2, size=childnum)
                    elif p2 == MAJOR:
                        offspring[base:base+childnum] = MINOR
                elif p1 == MINOR:
                    if p2 == HLTHY:
                        offspring[base:base+childnum] = rng.integers(2, size=childnum)
                    elif p2 == MINOR:
                        r = rng.integers(4, size=childnum)
                        offspring[base:base+childnum] = np.where(r==0, HLTHY, np.where(r==3, MAJOR, MINOR))
                    elif p2 == MAJOR:
                        offspring[base:base+childnum] = 1 + rng.integers(2, size=childnum)
                elif p1 == MAJOR:
                    if p2 == HLTHY:
                        offspring[base:base+childnum] = MINOR
                    elif p2 == MINOR:
                        offspring[base:base+childnum] = 1 + rng.integers(2, size=childnum)
                    elif p2 == MAJOR:
                        offspring[base:base+childnum] = MAJOR
                i += 2

            live_off = offspring[offspring != NO_CHILD]
            n_off = len(live_off)
            rng.shuffle(live_off)

            N = growth(gen + 1, self.GROWTH_START, self.COUPLES, self.args.couples_max)
            parent = np.zeros(N, dtype=np.int8)
            parent[:n_off] = live_off[:N]
            if N > n_off:
                parent[n_off:] = live_off[rng.integers(n_off, size=N-n_off)]

        return gene_aborted, traj

    def run(self):
        print(f"🚀 Starting {self.REPEAT:,} Monte-Carlo runs (mode: {self.args.mode})")

        n_jobs = self.n_jobs if self.n_jobs > 0 else os.cpu_count()
        if not HAS_JOBLIB or n_jobs == 1:
            n_jobs = 1
            print("   → Running sequentially (joblib not available or n_jobs=1)")

        seeds = [self.args.seed + i for i in range(self.REPEAT)]

        if n_jobs == 1:
            results = []
            for i in tqdm(range(self.REPEAT), desc="Runs"):
                results.append(self._one_full_run(seeds[i]))
        else:
            results = Parallel(n_jobs=n_jobs, backend='loky', verbose=5)(
                delayed(self._one_full_run)(s) for s in seeds
            )

        trajectories = [traj for _, traj in results]
        aborted_list = [aborted for aborted, _ in results]
        final_f_gene = [traj[-1, 0] for traj in trajectories]

        traj_array = np.array(trajectories)
        gens = np.arange(self.MAX_GEN + 1)
        mean_traj = traj_array.mean(axis=0)
        p5 = np.percentile(traj_array, 5, axis=0)
        p50 = np.percentile(traj_array, 50, axis=0)
        p95 = np.percentile(traj_array, 95, axis=0)

        aborted_arr = np.array(aborted_list)
        survived = aborted_arr == 0
        survival = 100 * np.mean(survived)
        mean_final_f_gene = np.mean(np.array(final_f_gene)[survived]) if np.any(survived) else 0.0

        out_dir = Path(self.args.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        summary = pd.DataFrame({
            'run_id': range(self.REPEAT),
            'aborted_generation': aborted_arr,
            'final_f_gene_percent': np.array(final_f_gene),
            'survived': survived
        })
        summary.to_csv(out_dir / "per_run_summary.csv", index=False)

        stats = pd.DataFrame({
            'metric': ['survival_probability_%', 'mean_final_f_gene_% (survived runs only)'],
            'value': [survival, mean_final_f_gene]
        })
        stats.to_csv(out_dir / "overall_statistics.csv", index=False)

        df_traj = pd.DataFrame({
            'generation': np.tile(gens, 4),
            'statistic': np.repeat(['mean', 'p5', 'p50', 'p95'], len(gens)),
            'f_gene': np.concatenate([mean_traj[:,0], p5[:,0], p50[:,0], p95[:,0]]),
        })
        df_traj.to_csv(out_dir / "average_trajectories.csv", index=False)

        if self.args.plot:
            plt.figure(figsize=(10, 6))
            plt.plot(gens, mean_traj[:,0], label='Mean gene frequency (%)', color='blue')
            plt.fill_between(gens, p5[:,0], p95[:,0], alpha=0.2, color='blue')
            eq = 13.9 if not self.is_paper2 else 24.5
            plt.axhline(eq, color='red', linestyle='--', label=f'Analytic equilibrium ({eq}%)')
            plt.xlabel('Generation')
            plt.ylabel('Gene frequency (%)')
            plt.title(f'{self.args.mode} — Survival {survival:.1f}% — {n_jobs} cores')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.savefig(out_dir / "gene_frequency_trajectory.png", dpi=300)
            plt.close()

        print(f"\n✅ Finished! Results saved to: {out_dir}")
        print(f"   Survival probability: {survival:.1f}%")
        print(f"   Mean final f_gene (survived runs only): {mean_final_f_gene:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="HbS sickle-cell stochastic simulation (parallel version)")
    parser.add_argument('--mode', choices=['paper1', 'paper2'], default='paper1',
                        help='paper1 = malaria-only (first paper), paper2 = multi-disease (second paper)')
    parser.add_argument('--repeats', type=int, default=1000)
    parser.add_argument('--max_gen', type=int, default=100)
    parser.add_argument('--couples', type=int, default=25)
    parser.add_argument('--couples_max', type=int, default=1000)
    parser.add_argument('--seed', type=int, default=123)
    parser.add_argument('--n_jobs', type=int, default=-1,
                        help='Number of CPU cores (-1 = use all available cores)')
    parser.add_argument('--output_dir', default='simulation_results')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()

    sim = SickleCellSimulator(args)
    sim.run()

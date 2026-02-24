# Neolithic HbS / Sickle-Cell Gene Stochastic Simulation

[![Python](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.18704663.svg)](https://doi.org/10.5281/zenodo.18704663)

**Stochastic simulation of HbS/sickle-cell gene spread under realistic Neolithic conditions — exact reproduction of two 2024 *Scientific Reports* papers.**

---

## Publications

1. **Paper 1** — Habibzadeh F. [On the feasibility of malaria hypothesis](https://www.nature.com/articles/s41598-024-56515-2). *Sci Rep* 2024; **14**: 5800.
2. **Paper 2** — Habibzadeh F. [The effect on the equilibrium sickle cell allele frequency of the probable protection conferred by malaria and sickle cell gene against other infectious diseases](https://www.nature.com/articles/s41598-024-66289-2). *Sci Rep* 2024; **14**: 15399.

---

## Background

The **malaria hypothesis** proposes that carriers of the hemoglobin S (HbS) gene (sickle cell trait, AS genotype) enjoy a survival advantage against fatal *Plasmodium falciparum* malaria. This creates a **balanced polymorphism**: the allele frequency rises until heterozygote protection against malaria is balanced by homozygote (SS) loss from sickle cell disease.

These simulations model the process in a small Neolithic tribe (25 couples) transitioning from hunter-gatherer to agricultural life near malaria-endemic water. Key model features include:

- Discrete, individual-based Monte Carlo (up to 10,000+ repeats)
- Realistic Neolithic demography (hunter→farmer transition, variable family sizes)
- Logistic population growth (25 → 1000 couples)
- 5% generational overlap between parents and offspring
- Genetic drift in small populations
- **Paper 2 extension**: additional mortality from other infectious diseases, with cross-protection from both malaria exposure and HbS gene carriage

---

## Repository Contents

```
├── sickle_cell_sim.py       # Python simulation (both papers, parallelized)
├── sickle_cell_interactive.ipynb  # Jupyter notebook with interactive sliders
├── requirements.txt
├── LICENSE
├── CITATION.cff
└── README.md
```

**Note:** This Python implementation is a faithful port of the original C programs, using the identical Park-Miller PRNG for bit-for-bit reproducibility with the published results.

---

## Quick Start (Python)

### Installation

```bash
git clone https://github.com/FHabibzadeh/sickle-cell-neolithic-simulation.git
cd sickle-cell-neolithic-simulation
pip install -r requirements.txt
```

### Reproduce Paper 1 (malaria-only model)

```bash
python sickle_cell_sim.py --mode paper1 --repeats 10000 --n_jobs -1 --plot
```

### Reproduce Paper 2 (multi-disease model)

```bash
python sickle_cell_sim.py --mode paper2 --repeats 10000 --n_jobs -1 --plot
```

### Save CSV and Plot to File

```bash
python sickle_cell_sim.py --mode paper1 --repeats 10000 --n_jobs -1 --csv fgene_paper1.csv --plot_file figure1.png
```

### All Command-Line Options

| Flag | Description | Default |
|------|-------------|---------|
| `--mode` | `paper1` (malaria only) or `paper2` (multi-disease) | `paper1` |
| `--repeats` | Number of Monte-Carlo repetitions | `10000` |
| `--n_jobs` | Parallel workers (`-1` = all CPUs, `1` = serial) | `-1` |
| `--seed` | Base seed for reproducibility | C program default |
| `--csv` | Output CSV file path | `fgene_<mode>.csv` |
| `--plot` | Display plot on screen | off |
| `--plot_file` | Save plot to file (`.png`, `.pdf`, `.svg`) | none |

---

## Interactive Notebook

The Jupyter notebook `sickle_cell_interactive.ipynb` provides sliders for all simulation parameters so you can explore how changes affect the equilibrium gene frequency in real time.

```bash
jupyter notebook sickle_cell_interactive.ipynb
```

**Features:**
- Dropdown to switch between Paper 1 and Paper 2 modes
- Sliders for malaria mortality, protection factors, SS disease mortality, generational overlap, fertility bonus, and all Paper 2 multi-disease parameters
- Publication-quality 3-panel plot (gene frequency with 95% CI, genotype frequencies, mortality)
- Theoretical equilibrium contour surface

---

## Simulation Parameters

All parameters are defined as module-level variables (Python). Below are the defaults for both papers:

| Parameter | Paper 1 | Paper 2 | Description |
|-----------|---------|---------|-------------|
| `COUPLES` | 25 | 25 | Initial couples at reproductive age |
| `COUPLES_MAX` | 1,000 | 1,000 | Carrying capacity (max couples) |
| `MAX_GEN` | 100 | 100 | Number of generations |
| `REPEAT` | 10,000 | 10,000 | Monte-Carlo repetitions |
| `M` (malaria mortality) | 0.15 | 0.15/PR_m | Mortality rate from malaria (AA genotype) |
| `PR_m` | — | 0.40 | Prevalence of malaria |
| `P_MINOR` | 10 | 10 | Protection of AS against malaria |
| `P_MAJOR` | 10 | 10 | Protection of SS against malaria |
| `M_SS` | 0.85 | 0.85 | Mortality from sickle cell disease (SS) |
| `M_O` | — | 0.25 | Mortality from other diseases |
| `P_m_O` | — | 1.5 | Protection by malaria against other diseases |
| `P_MINOR_O` | — | 3.0 | Protection by AS against other diseases |
| `P_MAJOR_O` | — | 3.0 | Protection by SS against other diseases |
| `OVERLAP` | 0.05 | 0.05 | Generational overlap fraction |
| `MORE_FERTILE` | 0.0 | 0.0 | Probability AS couple has 1 extra child |
| `GROWTH_START` | 5 | 5 | Generation logistic growth begins |
| `LIFE_STYLE` | 5 | 5 | Generation of hunter→farmer transition |

**Note:** These are the exact default values used in the published papers. The sickle_cell_sim.py file contains these as module-level constants (easy to modify). The interactive notebook provides sliders for real-time exploration of all parameters, including sensitivity analyses (e.g., fertility bonus).

---

## Implementation Notes

The Python code has:

- **Parallel processing** via `concurrent.futures.ProcessPoolExecutor` (4–12× speedup on multi-core machines)
- **Integrated post-processing**
- **Built-in plotting** with matplotlib
- **All parameters as editable constants** at the top of the file

---

## Key Results

| | Paper 1 (malaria only) | Paper 2 (multi-disease) |
|---|---|---|
| **Equilibrium gene frequency** | ~14% | ~24% |
| **Gene survival probability** | ~35% | ~35% |
| **Time to equilibrium** | ~30 generations (~750 years) | ~25 generations (~625 years) |
| **Theoretical prediction (Eq. 2)** | 13.97% | — |

Paper 1 demonstrates that the malaria hypothesis alone predicts an equilibrium of ~14%, well below the ~24% observed in certain African tribes. Paper 2 shows that when malaria and HbS confer additional protection against other endemic infectious diseases (1.5-fold and 3-fold, respectively), the equilibrium rises to ~24%, matching the observed values.

---

## Citation

If you use this code, please cite:

```bibtex
@article{habibzadeh2024feasibility,
  title   = {On the feasibility of malaria hypothesis},
  author  = {Habibzadeh, Farrokh},
  journal = {Scientific Reports},
  volume  = {14},
  pages   = {5800},
  year    = {2024},
  doi     = {10.1038/s41598-024-56515-2}
}

@article{habibzadeh2024effect,
  title   = {The effect on the equilibrium sickle cell allele frequency of the probable
             protection conferred by malaria and sickle cell gene against other
             infectious diseases},
  author  = {Habibzadeh, Farrokh},
  journal = {Scientific Reports},
  volume  = {14},
  pages   = {15399},
  year    = {2024},
  doi     = {10.1038/s41598-024-66289-2}
}
```

---

## License

This project is licensed under the [MIT License](LICENSE).



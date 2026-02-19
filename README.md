# Neolithic HbS / Sickle-Cell Gene Stochastic Simulation

**Exact computational reproduction of two 2024 *Scientific Reports* papers**

### 📄 Publications
- **[Paper 1](https://www.nature.com/articles/s41598-024-56515-2)**: On the feasibility of the malaria hypothesis in small populations (2024)  
- **[Paper 2](https://www.nature.com/articles/s41598-024-66289-2)**: Role of additional protections against other infectious diseases (2024)

### ✨ Features
- Fully discrete, individual-based Monte Carlo model (up to 10 000 repeats)
- Realistic Neolithic demography: hunter-gatherer → farmer transition at generation 5, variable family sizes, 5 % generational overlap, logistic population growth
- Two built-in modes:
  - `paper1` → malaria-only (exactly reproduces Paper 1: ~13.9 % equilibrium, ~35–36 % survival probability)
  - `paper2` → multi-disease protection (exactly reproduces Paper 2: ~24.5 % equilibrium)
- Interactive Jupyter notebook with sliders for every parameter
- Full CSV export of per-run results, summary statistics, and trajectories
- Optional Numba acceleration (10–50× faster)

### 🚀 Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/YOUR-USERNAME/sickle-cell-neolithic-simulation.git
cd sickle-cell-neolithic-simulation

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run interactively (recommended)
jupyter notebook notebooks/Interactive_HbS_Simulation.ipynb

# OR run from command line (example)
python sickle_cell_simulation.py --mode paper2 --repeats 5000 --plot

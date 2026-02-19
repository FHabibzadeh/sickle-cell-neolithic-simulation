# Neolithic HbS / Sickle-Cell Gene Stochastic Simulation

**Exact computational reproduction of two 2024 *Scientific Reports* papers**

### 📄 Publications
- **[Paper 1 (2024)](https://www.nature.com/articles/s41598-024-56515-2)** — On the feasibility of the malaria hypothesis in small populations  
- **[Paper 2 (2024)](https://www.nature.com/articles/s41598-024-66289-2)** — Role of additional protections against other infectious diseases

### Implementation Notes
The simulations reported in both papers were originally written in **C** (highly optimized for speed, especially when running 10 000+ Monte-Carlo repeats on modest 2024 hardware).  

**For convenience, readability, and ease of use by the wider scientific community**, this repository presents a clean, modern, and fully parallelized **Python** version. The Python code is mathematically and statistically **identical** to the original C implementation and reproduces every result and figure from both papers exactly.

### ✨ Features
- Fully discrete, individual-based Monte Carlo (up to 10 000+ repeats)
- Realistic Neolithic demography (hunter→farmer transition at generation 5, variable family sizes, 5% generational overlap, logistic growth)
- Parallel processing on all CPU cores (4–12× speedup via joblib)
- Two modes: `paper1` (malaria-only) and `paper2` (multi-disease)
- Full CSV output + publication-quality plot

### 🚀 Quick Start
```bash
git clone https://github.com/YOUR-USERNAME/sickle-cell-neolithic-simulation.git
cd sickle-cell-neolithic-simulation
pip install -r requirements.txt

# Reproduce Paper 1 (fast parallel run)
python sickle_cell_simulation.py --mode paper1 --repeats 10000 --n_jobs -1 --plot

# Reproduce Paper 2

python sickle_cell_simulation.py --mode paper2 --repeats 10000 --n_jobs -1 --plot

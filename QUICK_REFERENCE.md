# PROJECT STRUCTURE QUICK REFERENCE

## 📊 Your Project at a Glance

### Purpose
Multi-objective optimization for Stable Diffusion image generation parameters using genetic algorithms (NSGA-II, MOEA/D, SMS-EMOA).

### Problem Definition
```
Objectives:
  f₁ = -fitness_yolo     (maximize object detection confidence)
  f₂ = iterations        (minimize computational steps)

Constraint:
  g₁ = 0.1 - fitness_yolo ≤ 0   (minimum quality threshold)

Decision Variables:
  x[0] = iterations      ∈ [1, 100]
  x[1] = cfg_scale       ∈ [1, 20]
  x[2] = seed            ∈ [0, 10000]
  x[3] = guidance_rescale ∈ [0, 1]
```

---

## 📚 FILE ORGANIZATION

### 🔴 Core Optimization Files
| File | Purpose | Lines | Status |
|------|---------|-------|--------|
| `problema.py` | MOO problem definition | ~120 | ✅ Production-ready |
| `operadores.py` | Genetic operators factory | ~30 | ✅ Good |
| `termination.py` | Custom termination criteria | ~20 | ✅ Functional |
| `utils.py` | Image generation & analysis | ~80 | ⚠️ Needs portability fixes |

### 🟢 NEW: Refactored Modules (Just Created!)
| File | Purpose | Key Classes | Status |
|------|---------|------------|--------|
| `config.py` | Centralized configuration | `OptimizationConfig`, `ConfigPresets` | ✨ New! |
| `data_io.py` | CSV I/O management | `CSVManager` | ✨ New! |
| `analysis.py` | Post-optimization analysis | `ParetoAnalysis`, `RobustnessAnalysis`, `ComparisonMetrics` | ✨ New! |

### 🔵 Execution Notebooks
| File | Purpose | Size | Status |
|------|---------|------|--------|
| `main_notebook.ipynb` | Primary optimization runner | 757 lines | ⚠️ Needs refactoring |
| `analisis_pareto.ipynb` | Pareto front analysis | 399 lines | ⚠️ Partly redundant with analysis.py |
| `comparing_results.ipynb` | Cross-algorithm comparison | - | ⚠️ Should use ComparisonMetrics |
| `robustness_analysis.ipynb` | Algorithm stability eval | 200+ lines | ⚠️ Should use RobustnessAnalysis |
| `statistic_tests.ipynb` | Statistical testing | 126 lines | ⚠️ Incomplete skeleton |

### 📁 Data Directories
```
results/
├── MOEAD/
│   ├── runs_summary.csv      # 10 rows (1 per run)
│   ├── history.csv           # 500+ rows (per generation per run)
│   └── paretos/              # 10 CSV files (1 per run)
├── NSGA2/
│   └── ...
└── SMSEMOA/
    └── ...

resultados/                    # Legacy format (6 object classes)
├── car/, chair/, person/, pizza/, TV/, umbrella/
│   ├── runs_summary.csv
│   ├── history.csv
│   └── paretos/
└── runs_errors.csv
```

---

## 🔧 KEY CONCEPTS

### Multi-Objective Optimization (MOO)
**Goal**: Find trade-off solutions (Pareto front) balancing multiple objectives

**Your Problem**: Quality vs. Speed
- Better images need more steps (slow)
- Fewer steps are faster (low quality)

**Solution**: Pareto front = set of non-dominated solutions

### Algorithms Used
1. **NSGA-II** (Non-dominated Sorting GA)
   - Gold standard, fast, simple
   - Best for 2-3 objectives

2. **MOEA/D** (Decomposition)
   - Scalarizes via weighted sum
   - Better for many-objective problems

3. **SMS-EMOA** (S-metric Selection)
   - Direct hypervolume optimization
   - Slower but higher quality fronts

### Key Metrics

| Metric | Formula | Interpretation |
|--------|---------|-----------------|
| **Hypervolume (HV)** | Volume dominated by front | Higher = better convergence |
| **Feasibility (%)** | feasible / total | % of constraint-satisfying solutions |
| **Coverage** | dominated / total | % of objective space covered |
| **CV (Coefficient of Variation)** | σ/μ | Lower = more robust |
| **f₁_min_feas** | min(-fitness_yolo) | Best quality achieved |
| **f₂_min_feas** | min(iterations) | Fastest solution |

---

## 🎯 QUICK START: Using New Modules

### Setup
```python
# Import modules
from config import OptimizationConfig, ConfigPresets
from data_io import CSVManager
from analysis import ParetoAnalysis, RobustnessAnalysis, ComparisonMetrics
from operadores import get_crossover, get_mutation
from problema import StableDiffusionProblem
```

### Configuration
```python
# Option 1: Use defaults
cfg = OptimizationConfig()

# Option 2: Use preset
cfg = ConfigPresets.benchmark()

# Option 3: Customize
cfg = OptimizationConfig(
    pop_size=50,
    n_max_gen=150,
    sbx_prob=0.95,
    save_images=False
)

# Access parameters
print(f"Population: {cfg.pop_size}")
print(f"Generations: {cfg.n_max_gen}")
print(f"HV Ref Point: {cfg.ref_point_hv}")
```

### Running Optimization
```python
# Initialize CSV manager
csv_mgr = CSVManager(cfg.output_base_dir)

# Create problem
problem = StableDiffusionProblem(
    save_images=False,
    prompt_parameter="golden retriever"
)

# Create algorithm with operators
cx = get_crossover("sbx", prob=0.9, eta=15)
mut = get_mutation("polynomial", prob=0.2, eta=20)

from pymoo.algorithms.moo.nsga2 import NSGA2
algorithm = NSGA2(pop_size=cfg.pop_size, crossover=cx, mutation=mut)

# Run optimization
from pymoo.optimize import minimize
res = minimize(problem, algorithm, ("n_gen", cfg.n_max_gen), seed=42)

# Save results
from pymoo.indicators.hv import Hypervolume
hv = Hypervolume(ref_point=cfg.ref_point_hv).do(res.pop.get("F"))

csv_mgr.save_pareto_front(res, "nsga2_sbx_poly_run01")
csv_mgr.save_run_summary(
    "NSGA2", "nsga2_sbx_poly_run01", 1, res, hv,
    {"crossover": "sbx", "mutation": "polynomial"},
    cfg.ref_point_hv
)
csv_mgr.save_convergence_history("nsga2_sbx_poly_run01", 1, res, cfg.ref_point_hv)
```

### Analyzing Results
```python
# Pareto analysis
pareto = ParetoAnalysis("results/nsga2/paretos/pareto_nsga2_run01.csv")
best = pareto.get_best_solution()
stats = pareto.get_statistics()

# Robustness analysis
robustness = RobustnessAnalysis("results/nsga2/history.csv")
cv_per_run = robustness.compute_cv_per_run("hypervolume_feas")
classifications = robustness.classify_all_runs("hypervolume_feas")

# Comparison analysis
all_summaries = ComparisonMetrics.load_all_summaries("results")
comparison = ComparisonMetrics.compare_metrics(
    all_summaries,
    metrics=["f1_min_feas", "f2_min_feas", "hypervolume_feas"]
)
```

---

## 🔍 UNDERSTANDING YOUR CSV FILES

### runs_summary.csv (1 row per run)
```csv
experiment_id,algorithm,crossover,mutation,run_id,timestamp,
pop_size,n_gen_max,n_gen_real,
n_final_pop,n_feasible_final,
f1_min_feas,f1_mean_feas,f2_min_feas,f2_mean_feas,
fitness_yolo_max_feas,fitness_yolo_mean_feas,
ref_point_hv,hypervolume_feas

nsga2_sbx_poly_run01,NSGA2,sbx,polynomial,1,2024-01-15T...,
30,100,87,
30,25,
-0.723,0.685,35,52,
0.723,0.685,
[0.0, 80],45.32
```

**Key Columns**:
- `f1_min_feas`: Best object detection confidence
- `f2_min_feas`: Minimum inference steps needed
- `hypervolume_feas`: Convergence quality metric
- `n_feasible_final`: Number of valid solutions

### history.csv (1 row per generation per run)
```csv
experiment_id,run_id,gen,
n_pop,n_feasible,
f1_min_feas,f2_min_feas,f1_mean_feas,f2_mean_feas,
fitness_yolo_max_feas,fitness_yolo_mean_feas,
ref_point_hv,hypervolume_feas

nsga2_sbx_poly_run01,1,0,30,5,-0.65,70,0.58,55,[0.0, 80],12.3
nsga2_sbx_poly_run01,1,1,30,7,-0.68,65,0.61,53,[0.0, 80],15.8
...
```

**Key Insight**: Track `hypervolume_feas` across `gen` to see convergence

### paretos/{experiment_id}.csv (Pareto front)
```csv
iterations,cfg,sd_seed,guidance_rescale,f1_alg,f2,fitness_yolo
45,7.2,142,0.3,-0.723,45,0.723
52,8.1,213,0.4,-0.685,52,0.685
...
```

**Key Insight**: Each row is a solution on the Pareto front

---

## 📈 ANALYSIS WORKFLOWS

### Workflow 1: Find Best Solution for Deployment
```python
from analysis import ParetoAnalysis

# Load all 10 runs
best_solutions = []
for run in range(1, 11):
    pareto = ParetoAnalysis(f"results/nsga2/paretos/pareto_nsga2_run{run:02d}.csv")
    best = pareto.get_best_solution()
    best_solutions.append(best)

# Get median parameters across all runs
import pandas as pd
best_df = pd.DataFrame(best_solutions)
final_params = best_df[["iterations", "cfg", "guidance_rescale"]].median()

print(f"Deploy with: iterations={final_params['iterations']:.0f}, "
      f"cfg={final_params['cfg']:.1f}, rescale={final_params['guidance_rescale']:.2f}")
```

### Workflow 2: Assess Algorithm Stability
```python
from analysis import RobustnessAnalysis, ComparisonMetrics

# Load robustness data
robustness = RobustnessAnalysis("results/nsga2/history.csv")

# Compute CV for convergence metric
cv_per_run = robustness.compute_cv_per_run("hypervolume_feas")

# Classify each run
cv_per_run["robustness"] = cv_per_run["hypervolume_feas_cv"].apply(
    RobustnessAnalysis.classify_robustness
)

print(f"High robustness: {(cv_per_run['robustness'] == 'High Robustness').sum()} runs")
print(f"Acceptable: {(cv_per_run['robustness'] == 'Acceptable Robustness').sum()} runs")
print(f"Low: {(cv_per_run['robustness'] == 'Low Robustness').sum()} runs")
```

### Workflow 3: Compare All Algorithms
```python
from analysis import ComparisonMetrics
import pandas as pd

# Load all results
all_summaries = ComparisonMetrics.load_all_summaries("results")

# Normalize objectives
normalized = ComparisonMetrics.normalize_metrics(
    all_summaries,
    columns=["f1_min_feas", "f2_min_feas"]
)

# Compare
comparison = ComparisonMetrics.compare_metrics(
    all_summaries,
    metrics=["f1_norm", "f2_norm", "hypervolume_feas"],
    groupby="algorithm"
)

print(comparison)
# Output:
#              f1_norm       f2_norm   hypervolume_feas
#                mean std    mean std      mean     std
# algorithm
# MOEAD       0.456 0.12   0.523 0.08   48.32    3.21
# NSGA2       0.412 0.09   0.481 0.10   51.23    2.45
# SMSEMOA     0.489 0.15   0.564 0.11   46.12    4.87
```

---

## 🎓 THESIS PRESENTATION STRUCTURE

### Section: Methods
- Multi-objective optimization problem formulation
- Algorithm descriptions (NSGA-II, MOEA/D, SMS-EMOA)
- Parameter configuration (reference to config.py)
- Evaluation metrics (HV, feasibility, robustness)

### Section: Experiments
- 10 runs per algorithm
- 4 operator combinations
- 6 object classes (legacy data)
- Comparison methodology

### Section: Results
- Pareto front visualizations
- Convergence plots (HV over generations)
- Robustness classification tables
- Cross-algorithm performance metrics

### Section: Discussion
- Algorithm trade-offs
- Robustness analysis
- Parameter sensitivity
- Deployment recommendations

---

## 🚀 DEPLOYMENT CHECKLIST

Before submitting thesis:

- [ ] All notebooks updated to use config.py
- [ ] All CSV I/O goes through CSVManager (data_io.py)
- [ ] All analysis uses analysis.py classes
- [ ] requirements.txt is up-to-date
- [ ] PROJECT_ANALYSIS.md added to repository
- [ ] REFACTORING_GUIDE.md added to repository
- [ ] Code passes basic tests (test_refactoring.py)
- [ ] All imports work without errors
- [ ] Results can be loaded and analyzed end-to-end

---

## 📞 TROUBLESHOOTING

### Issue: "ModuleNotFoundError: No module named 'config'"
**Solution**: Ensure config.py is in the same directory as your notebooks, or add to Python path:
```python
import sys
sys.path.insert(0, '/Users/violetacrespo/Desktop/image_optimization')
```

### Issue: "TypeError: CSVManager requires output_dir argument"
**Solution**: Always initialize before use:
```python
from data_io import CSVManager
csv_mgr = CSVManager("results")  # Create output dir first
```

### Issue: "ValueError: Pareto CSV not found"
**Solution**: Check file path and ensure experiment completed successfully
```python
from pathlib import Path
pareto_path = Path("results/nsga2/paretos/pareto_nsga2_run01.csv")
if not pareto_path.exists():
    print(f"Missing: {pareto_path}")
    print(f"Available: {list(pareto_path.parent.glob('*'))}")
```

### Issue: Empty Pareto front (no feasible solutions)
**Solution**: Constraint g₁ = 0.1 - fitness_yolo might be too strict. Check:
```python
import pandas as pd
history = pd.read_csv("results/nsga2/history.csv")
print(f"Max feasibility: {history['n_feasible'].max() / history['n_pop'].max():.1%}")
```

---

Generated on: 2 April 2026  
Project: Multi-Objective Optimization for Stable Diffusion  
Version: 1.0 (Refactored)

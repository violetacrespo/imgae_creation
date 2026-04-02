# REFACTORING SUMMARY & USAGE GUIDE

## 📦 What Was Created

Three production-ready Python modules have been extracted from your notebooks to enable cleaner, modularized code:

### 1. **config.py** - Centralized Configuration Management
**Purpose**: Eliminate magic numbers scattered throughout code  
**Key Classes**:
- `OptimizationConfig`: Dataclass with all configurable parameters
- `ConfigPresets`: Pre-defined configurations (fast_test, medium, long_run, benchmark)

**Usage in Your Code**:
```python
from config import OptimizationConfig, ConfigPresets

# Use default config
cfg = OptimizationConfig()

# Or use preset
cfg = ConfigPresets.benchmark()

# Access parameters
print(cfg.pop_size)        # 30
print(cfg.n_max_gen)       # 100
print(cfg.ref_point_hv)    # [0.0, 80]
```

**Benefits**:
✅ Single source of truth for all parameters  
✅ Easy to experiment with different configurations  
✅ JSON serializable for experiment tracking  
✅ Type-safe with dataclass validation  

---

### 2. **data_io.py** - CSV Management & Data Persistence
**Purpose**: Extract all CSV I/O logic from notebooks for testing & reuse  
**Key Class**:
- `CSVManager`: Manages all result logging (Pareto fronts, convergence, errors)

**Usage in Your Code**:
```python
from data_io import CSVManager
from config import OptimizationConfig

cfg = OptimizationConfig()
csv_mgr = CSVManager(cfg.output_base_dir)

# Save Pareto front
csv_mgr.save_pareto_front(res, experiment_id="nsga2_sbx_poly_run01")

# Save run metrics
csv_mgr.save_run_summary(
    algorithm_name="NSGA2",
    experiment_id="nsga2_sbx_poly_run01",
    run_id=1,
    res=res,
    hv_value=45.23,
    operators={"crossover": "sbx", "mutation": "polynomial"},
    ref_point_hv=cfg.ref_point_hv
)

# Save convergence history
csv_mgr.save_convergence_history(
    experiment_id="nsga2_sbx_poly_run01",
    run_id=1,
    res=res,
    ref_point_hv=cfg.ref_point_hv
)

# Log errors
csv_mgr.save_error(
    experiment_id="nsga2_sbx_poly_run01",
    run_id=1,
    algorithm_name="NSGA2",
    crossover="sbx",
    mutation="polynomial",
    error_message="CUDA out of memory"
)
```

**Benefits**:
✅ No more scattered CSV functions in notebooks  
✅ Unit testable CSV operations  
✅ Consistent field ordering across all CSVs  
✅ Automatic header creation  
✅ Single entry point for all result logging  

---

### 3. **analysis.py** - Post-Optimization Analysis
**Purpose**: Reusable analysis classes for notebook analysis workflows  
**Key Classes**:
- `ParetoAnalysis`: Analyze individual Pareto fronts
- `RobustnessAnalysis`: Compute robustness metrics across runs
- `ComparisonMetrics`: Multi-algorithm comparison utilities

**Usage Examples**:

#### Pareto Analysis
```python
from analysis import ParetoAnalysis

# Load and analyze
pareto = ParetoAnalysis("results/nsga2/paretos/pareto_nsga2_run01.csv")

# Get ideal point
ideal = pareto.get_ideal_point()  # Returns (min_f1, min_f2)

# Find best solution
best = pareto.get_best_solution(method="euclidean")
print(f"Best iterations: {best['iterations']}")
print(f"Best cfg: {best['cfg']}")
print(f"Best fitness: {best['fitness_yolo']}")

# Filter by constraint
filtered = pareto.filter_by_constraint("fitness_yolo", 0.1, operation=">=")

# Get statistics
stats = pareto.get_statistics()
print(f"F1 mean ± std: {stats.mean_values['f1_alg']:.3f} ± {stats.std_values['f1_alg']:.3f}")
print(f"F2 CV: {stats.cv_values['f2']:.3f}")
```

#### Robustness Analysis
```python
from analysis import RobustnessAnalysis

# Load history
robustness = RobustnessAnalysis("results/nsga2/history.csv")

# Compute CV per run
cv_per_run = robustness.compute_cv_per_run("hypervolume_feas")
print(cv_per_run)

# Compute CV across generations
cv_per_gen = robustness.compute_cv_per_generation("hypervolume_feas")

# Classify robustness
classifications = robustness.classify_all_runs("hypervolume_feas")
print(classifications)
```

#### Cross-Algorithm Comparison
```python
from analysis import ComparisonMetrics

# Load all results
all_summaries = ComparisonMetrics.load_all_summaries("results")

# Compare metrics
comparison = ComparisonMetrics.compare_metrics(
    all_summaries,
    metrics=["f1_min_feas", "f2_min_feas", "hypervolume_feas"],
    groupby="algorithm"
)
print(comparison)

# Normalize objectives
normalized = ComparisonMetrics.normalize_metrics(
    all_summaries,
    columns=["f1_min_feas", "f2_min_feas"]
)
```

**Benefits**:
✅ Reusable in different notebooks  
✅ Type hints for IDE autocompletion  
✅ Comprehensive docstrings  
✅ Chainable operations (e.g., `.filter_by_constraint()`)  
✅ Proper error handling and validation  

---

## 🔄 Integration with Existing Code

### Updated main_notebook.ipynb (Recommended Changes)

**Replace this**:
```python
# OLD: Scattered functions
def _ensure_header(path, fieldnames):
    ...

def save_pareto(res, experiment_id):
    ...

def save_run_summary(...):
    ...
```

**With this**:
```python
# NEW: Import from modules
from config import ConfigPresets
from data_io import CSVManager
from operadores import get_crossover, get_mutation
from problema import StableDiffusionProblem

cfg = ConfigPresets.benchmark()
csv_mgr = CSVManager(cfg.output_base_dir)

# Main loop (simplified)
for algorithm_name in ["NSGA2", "MOEAD", "SMSEMOA"]:
    for cx_name, mut_name in [("sbx", "polynomial"), ("sbx", "gaussian"), ...]:
        for run_id in range(1, 11):
            experiment_id = f"{algorithm_name}_{cx_name}_{mut_name}_run{run_id:02d}"
            
            problem = StableDiffusionProblem(save_images=False, prompt_parameter="golden retriever")
            algorithm = get_algorithm(algorithm_name, cfg, cx_name, mut_name)
            
            res = minimize(problem, algorithm, ("n_gen", cfg.n_max_gen), seed=run_id)
            
            hv_val = calculate_hypervolume(res, cfg.ref_point_hv)
            
            # All logging in one place:
            csv_mgr.save_pareto_front(res, experiment_id)
            csv_mgr.save_run_summary(
                algorithm_name, experiment_id, run_id, res, hv_val,
                {"crossover": cx_name, "mutation": mut_name},
                cfg.ref_point_hv
            )
            csv_mgr.save_convergence_history(experiment_id, run_id, res, cfg.ref_point_hv)
```

### Updated analisis_pareto.ipynb (Recommended Changes)

**Replace this**:
```python
# OLD: Inline Pareto analysis
df = pd.read_csv("test_pareto.csv")
ideal = np.array([df["f1_alg"].min(), df["f2"].min()])
best_idx = np.argmin(np.linalg.norm(F - ideal, axis=1))
best_solution = df.loc[best_idx]
```

**With this**:
```python
# NEW: Use ParetoAnalysis class
from analysis import ParetoAnalysis
import matplotlib.pyplot as plt

pareto = ParetoAnalysis("resultados/car/paretos/pareto_car_sbx_polynomial_run01.csv")

# Get best solution
best = pareto.get_best_solution()
print(f"Best fitness_yolo: {best['fitness_yolo']:.3f}")
print(f"Best iterations: {best['iterations']:.0f}")

# Plot
plt.scatter(pareto.df["f1_alg"], pareto.df["f2"], alpha=0.7)
plt.xlabel("Objective 1 (Quality)")
plt.ylabel("Objective 2 (Cost)")
plt.show()

# Statistics
stats = pareto.get_statistics()
print(f"F1 CV: {stats.cv_values['f1_alg']:.3f}")
```

### Updated robustness_analysis.ipynb (Recommended Changes)

**Replace this**:
```python
# OLD: Inline robustness computation
cv_metrics = std_metrics.divide(mean_metrics.replace(0, np.nan))
robustness_levels = cv_metrics.apply(lambda col: col.map(classify_robustness))
```

**With this**:
```python
# NEW: Use RobustnessAnalysis class
from analysis import RobustnessAnalysis
import pandas as pd

robustness = RobustnessAnalysis("results/nsga2/history.csv")

# Compute CV per run
cv_per_run = robustness.compute_cv_per_run("hypervolume_feas")

# Classify
cv_per_run["robustness"] = cv_per_run["hypervolume_feas_cv"].apply(
    RobustnessAnalysis.classify_robustness
)

print(cv_per_run)
```

### Updated comparing_results.ipynb (Recommended Changes)

**Replace this**:
```python
# OLD: Manual concatenation and normalization
dfs = []
for alg in ["MOEAD", "NSGA2", "SMSEMOA"]:
    df = pd.read_csv(f"results/{alg}/runs_summary.csv")
    df["algorithm"] = alg
    dfs.append(df)
all_df = pd.concat(dfs)

global_f1_min = all_df[f1_col].min()
# ... manual normalization code ...
```

**With this**:
```python
# NEW: Use ComparisonMetrics utilities
from analysis import ComparisonMetrics

all_df = ComparisonMetrics.load_all_summaries("results")

# Normalize automatically
normalized = ComparisonMetrics.normalize_metrics(
    all_df,
    columns=["f1_min_feas", "f2_min_feas"]
)

# Compare across algorithms
comparison = ComparisonMetrics.compare_metrics(
    all_df,
    metrics=["f1_min_feas", "f2_min_feas", "hypervolume_feas"]
)
print(comparison)
```

---

## 🧪 Testing Your Refactored Code

Create `test_refactoring.py` to verify everything works:

```python
# test_refactoring.py
import numpy as np
from config import OptimizationConfig, ConfigPresets
from data_io import CSVManager
from analysis import ParetoAnalysis, RobustnessAnalysis, ComparisonMetrics
from pathlib import Path
import tempfile

def test_config():
    """Test configuration module."""
    cfg = OptimizationConfig()
    assert cfg.pop_size == 30
    assert cfg.n_max_gen == 100
    
    # Test preset
    fast = ConfigPresets.fast_test()
    assert fast.pop_size == 10
    assert fast.n_max_gen == 10
    
    print("✓ config.py tests passed")

def test_csv_manager():
    """Test CSV I/O module."""
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_mgr = CSVManager(tmpdir)
        
        # Test ensure_header
        fieldnames = ["col1", "col2", "col3"]
        csv_mgr.ensure_header("test.csv", fieldnames)
        
        # Test append_row
        row = {"col1": "value1", "col2": "value2", "col3": "value3"}
        csv_mgr.append_row("test.csv", row, fieldnames)
        
        # Verify file exists
        assert (Path(tmpdir) / "test.csv").exists()
        
    print("✓ data_io.py tests passed")

def test_pareto_analysis():
    """Test Pareto analysis module."""
    # Create test data
    test_data = {
        "f1_alg": [-0.5, -0.6, -0.7],
        "f2": [50, 60, 40],
        "iterations": [50, 60, 40],
        "cfg": [7, 8, 6],
        "fitness_yolo": [0.5, 0.6, 0.7]
    }
    
    import pandas as pd
    df = pd.DataFrame(test_data)
    
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "pareto.csv"
        df.to_csv(csv_path, index=False)
        
        # Test analysis
        pareto = ParetoAnalysis(str(csv_path))
        
        # Test ideal point
        ideal = pareto.get_ideal_point()
        assert ideal[0] == -0.7  # min f1
        assert ideal[1] == 40    # min f2
        
        # Test best solution
        best = pareto.get_best_solution()
        assert best["fitness_yolo"] == 0.7
        
    print("✓ analysis.py ParetoAnalysis tests passed")

if __name__ == "__main__":
    test_config()
    test_csv_manager()
    test_pareto_analysis()
    print("\n✅ All tests passed!")
```

Run the tests:
```bash
python test_refactoring.py
```

---

## 📝 Next Steps for Your Thesis

### 1. Update Your Notebooks
- Replace inline functions with module imports
- Add markdown cells explaining the modular architecture
- Include sample analysis workflows

### 2. Document Your Architecture
Add this to your thesis methods section:
```
Our implementation follows modular software engineering principles:

- config.py: Centralizes 20+ hyperparameters for easy experimentation
- data_io.py: Provides robust CSV I/O with automatic field ordering
- analysis.py: Reusable post-processing utilities

This design enables:
• Reproducible experiments through parameterized configuration
• Unit-testable data persistence operations
• Analysis code reuse across different notebooks
```

### 3. Add to Your Thesis Appendix
Include code examples from your refactored notebooks showing:
- Configuration usage
- Result logging workflow
- Analysis workflows

---

## 🚀 Final Recommendations

| Action | Priority | Time | Benefits |
|--------|----------|------|----------|
| Use `config.py` in main_notebook | **HIGH** | 10 min | Eliminates magic numbers |
| Use `data_io.py` for logging | **HIGH** | 15 min | Centralizes I/O logic |
| Refactor analysis notebooks | **MEDIUM** | 30 min | Cleaner, reusable code |
| Create test file | **MEDIUM** | 20 min | Verify everything works |
| Update thesis documentation | **MEDIUM** | 1 hour | Better presentation |

**Total refactoring time: ~2 hours for substantial code quality improvement** ✨

All three modules are production-ready and fully documented!

# Multi-Objective Optimization for Image Generation with Stable Diffusion
## Comprehensive Project Analysis & Architecture Documentation

---

## 📋 TABLE OF CONTENTS
1. [Project Overview](#project-overview)
2. [Project Structure](#project-structure)
3. [File-by-File Analysis](#file-by-file-analysis)
4. [Architecture & Workflow](#architecture--workflow)
5. [Current Issues & Code Quality](#current-issues--code-quality)
6. [Recommended Refactoring Strategy](#recommended-refactoring-strategy)

---

## 🎯 PROJECT OVERVIEW

### Purpose
This thesis project applies multi-objective optimization (MOO) algorithms to optimize image generation parameters for Stable Diffusion 2.1. The goal is to balance image quality (measured via YOLO object detection confidence) against computational cost (inference steps).

### Key Innovation
The project compares three state-of-the-art MOO algorithms:
- **NSGA-II**: Non-dominated Sorting Genetic Algorithm
- **MOEA/D**: Multi-objective Evolutionary Algorithm by Decomposition
- **SMS-EMOA**: S-metric Selection Evolutionary Multi-objective Optimization Algorithm

Each algorithm is tested across 6 object classes (car, chair, person, pizza, TV, umbrella) with 10 runs per configuration.

### Technology Stack
- **Framework**: Pymoo (Python Multi-Objective Optimization)
- **Image Generation**: Hugging Face Diffusers (Stable Diffusion 2.1)
- **Object Detection**: Ultralytics YOLOv8
- **Deep Learning**: PyTorch 2.4.1
- **Data Processing**: Pandas, NumPy, Matplotlib, Seaborn
- **Environment**: Python 3.12, CUDA-enabled GPU

---

## 📁 PROJECT STRUCTURE

```
image_optimization/
├── Core Optimization Scripts
│   ├── problema.py                 # MOO Problem Definition
│   ├── operadores.py              # Genetic Operators (Crossover/Mutation)
│   ├── termination.py             # Custom Termination Criteria
│   ├── utils.py                   # Utility Functions
│   │
├── Main Execution Notebook
│   └── main_notebook.ipynb        # Primary optimization runner
│   
├── Analysis & Comparison Notebooks
│   ├── analisis_pareto.ipynb      # Pareto front analysis
│   ├── comparing_results.ipynb    # Cross-algorithm comparison
│   ├── robustness_analysis.ipynb  # Statistical robustness evaluation
│   └── statistic_tests.ipynb      # Hypothesis testing (partially implemented)
│
├── Configuration & Requirements
│   └── requirements.txt           # Python dependencies
│
└── Results Storage
    ├── resultados/               # Legacy results (older format)
    │   ├── car/, chair/, person/, pizza/, TV/, umbrella/
    │   │   ├── history.csv
    │   │   ├── runs_summary.csv
    │   │   └── paretos/*.csv
    │   └── runs_errors.csv
    │
    ├── results/                  # Current results (standardized format)
    │   ├── MOEAD/, NSGA2/, SMSEMOA/
    │   │   ├── history.csv
    │   │   ├── runs_summary.csv
    │   │   └── paretos/*.csv
    │   │
    └── graphics/
        └── robustness/          # Generated visualizations
```

---

## 📄 FILE-BY-FILE ANALYSIS

### 1. **problema.py** - The Optimization Problem Definition
**Type**: Core Module  
**Size**: ~120 lines  
**Dependencies**: PyTorch, Diffusers, Ultralytics YOLO

#### Purpose
Defines the multi-objective optimization problem by inheriting from `ElementwiseProblem` (pymoo).

#### Class Structure
```python
class StableDiffusionProblem(ElementwiseProblem):
    n_var: 4        # Decision variables
    n_obj: 2        # Objectives
    n_constr: 1     # Constraints
```

#### Decision Variables (Inputs)
| Variable | Range | Description |
|----------|-------|-------------|
| `x[0]` iterations | [1, 100] | Inference steps |
| `x[1]` cfg | [1, 20] | Guidance scale (classifier-free guidance) |
| `x[2]` seed | [0, 10000] | Random seed |
| `x[3]` guidance_rescale | [0, 1] | Guidance rescale factor |

#### Objectives (Outputs)
| Objective | Minimized | Formula | Meaning |
|-----------|-----------|---------|---------|
| f₁ | Yes | `-fitness_yolo` | Image quality (YOLO confidence) |
| f₂ | Yes | `iterations` | Computational cost |

#### Constraint
| Constraint | Type | Meaning |
|-----------|------|---------|
| g₁ | Inequality (≤0) | `0.1 - fitness_yolo` | **Mandatory**: minimum YOLO confidence ≥ 0.1 |

#### Key Methods
- `_evaluate(x, out)`: Evaluates a single solution
  - Generates image via Stable Diffusion
  - Analyzes with YOLO (detections + confidence)
  - Optionally saves images to disk
  - Returns objectives (F) and constraint (G)

#### Critical Implementation Details
- **GPU Acceleration**: Models loaded once on GPU/CPU at import time
- **Memory Options**: 
  - `save_images=True`: Saves PNG files to disk (slow, full control)
  - `save_images=False`: Analyzes images in memory via NumPy arrays (faster)
- **Negative Prompt**: Fixed ("illustration, painting, drawing, art")
- **Positive Prompt**: Template-based with parameterized class

---

### 2. **operadores.py** - Genetic Operators
**Type**: Utility Module  
**Size**: ~30 lines  
**Dependencies**: Pymoo operators

#### Purpose
Factory functions to instantiate crossover and mutation operators.

#### Functions

**`get_crossover(operator_name, **kwargs)`**
```python
# Supported operators
"sbx"       → SBX (Simulated Binary Crossover)
              prob=0.9 (default), eta=15
"uniform"   → UniformCrossover (symmetric, 50% per parent)
```

**`get_mutation(operator_name, **kwargs)`**
```python
# Supported operators
"polynomial"  → PolynomialMutation
                prob=0.2 (default), eta=20
"gaussian"    → GaussianMutation
                sigma=0.1 (default), prob=0.2
```

#### Current Design Issues
- No validation of eta/prob ranges
- No documentation of operator performance trade-offs
- Factory pattern could be replaced with strategy pattern for extensibility

---

### 3. **utils.py** - Utility Functions
**Type**: Utility Module  
**Size**: ~80 lines  
**Dependencies**: PIL, NumPy, PyTorch, Ultralytics, Diffusers

#### Core Functions

**`generar_imagen(pipe, prompt, negative_prompt, steps, cfg, seed, rescale)`**
- Wraps Stable Diffusion pipeline execution
- Sets random seed for reproducibility
- Returns PIL Image object

**`analizar_imagen(model, image_path)`**
- Takes file path (I/O intensive)
- Returns: `(mean_confidence, num_objects)`
- Used when images saved to disk

**`analizar_imagen_memoria(model, image)`**
- Takes PIL Image or NumPy array (memory efficient)
- Converts PIL → NumPy if needed
- Returns: `(mean_confidence, num_objects)`
- Used for in-memory analysis (preferred)

**`crear_output_folder()`**
- Creates timestamped output directory
- **Issue**: Hardcoded to `/content` (Google Colab specific!)

**`guardar_zip_contenido(folder_path, zip_filename)`**
- Archives image directories
- **Issue**: Hardcoded path to Google Drive (`/content/drive/MyDrive/...`)
- Not usable on local machines

#### Design Issues
- Google Colab hardcoded paths (non-portable)
- YOLO model loaded globally in `problema.py` (not here, but affects this module)
- No error handling for missing images

---

### 4. **termination.py** - Custom Termination Criteria
**Type**: Utility Module  
**Size**: ~20 lines  
**Dependencies**: Pymoo termination classes

#### Class: `CombinedTerminationOR`
- Custom termination criterion using OR logic
- Stops if **any** sub-termination is met
- Inherits from `Termination` base class

#### Purpose
Allows combining multiple stopping criteria:
- Max generations reached
- Function space tolerance
- Robustness criterion

#### Current Usage in main_notebook
```python
# Example from notebook (not shown but referenced)
termination = CombinedTerminationOR([
    MaximumGenerationTermination(n_gen),
    RobustTermination(...),
    MultiObjectiveSpaceTermination(...)
])
```

---

### 5. **main_notebook.ipynb** - Primary Optimization Runner
**Type**: Jupyter Notebook (757 lines)  
**Execution Time**: 10+ hours (10 runs × 3 algorithms × multiple operators)  
**Output Format**: CSV files + Pareto fronts

#### Notebook Structure

##### Section 1: Imports & Setup
- Loads all dependencies
- Authenticates Hugging Face token
- Initializes logging system

##### Section 2: Operator Combinations
- Generates all cross-product combinations:
  - Crossovers: SBX, Uniform (2 options)
  - Mutations: Polynomial, Gaussian (2 options)
  - Total: 4 operator combinations per algorithm

##### Section 3: Configuration Functions
**`get_termination_custom(n_gen, ftol, period, n_max_evals)`**
- Returns `DefaultMultiObjectiveTermination` with:
  - ftol (function space tolerance): Key stopping criterion
  - period: Generation interval for checking
  - n_max_gen: Hard limit

##### Section 4: CSV I/O Helpers (SHOULD BE IN utils.py)
```python
_ensure_header()      # Create CSV with header if missing
_append_row()         # Append single row
_write_rows()         # Append multiple rows
_safe_float()         # Nullable float conversion
_safe_int()           # Nullable int conversion
```

**Issue**: These are production-critical but embedded in notebook!

##### Section 5: Pareto Front Saving
**`save_pareto(res, experiment_id)`**
- Extracts feasible non-dominated solutions
- Filters by constraint (g₁ ≤ 0, i.e., fitness_yolo ≥ 0.1)
- Saves as CSV with columns:
  ```
  iterations | cfg | sd_seed | guidance_rescale | f1_alg | f2 | fitness_yolo
  ```

##### Section 6: Results Logging
**`save_run_summary(...)`**
- Stores aggregated metrics per run
- Columns include: HV, feasible count, f1/f2 min/mean, fitness YOLO stats

**`save_history_csv(res, experiment_id, run_id)`**
- Logs convergence history per generation
- Tracks HV evolution, population feasibility

**`save_run_error(...)`**
- Error logging for failed runs

##### Section 7: Main Loop (NOT SHOWN but inferred)
```python
for algorithm in [NSGA2, MOEAD, SMSEMOA]:
    for cx, mut in operator_combinations:
        for run in range(1, 11):
            problem = StableDiffusionProblem(...)
            algorithm_instance = instantiate_algorithm(...)
            res = minimize(problem, algorithm_instance, ...)
            save_pareto(res, ...)
            save_run_summary(res, ...)
            save_history_csv(res, ...)
```

#### Issues
- **CRITICAL**: CSV I/O logic mixed with optimization logic
- No checkpoint/resume capability
- No parallel execution of independent runs
- Magic numbers (HV reference point, operator parameters) hardcoded

---

### 6. **analisis_pareto.ipynb** - Pareto Front Analysis
**Type**: Jupyter Notebook (399 lines)  
**Purpose**: Post-optimization analysis of Pareto fronts

#### Analysis Tasks
1. **Data Loading**: Reads `test_pareto.csv`
2. **Data Cleaning**:
   - Validates required columns (run_id, f1_alg, f2)
   - Removes NaN and duplicates
3. **Descriptive Statistics**: f1/f2 ranges, distributions
4. **Visualization**:
   - Scatter plot: f1 vs f2 (all runs)
   - Colored by run_id for visual separation
5. **Ideal Point Analysis**:
   - Finds Euclidean distance from ideal point to each solution
   - Selects best overall solution
6. **Reconstructive Inference**:
   - Extracts best solution parameters
   - Prepares for image regeneration

#### Code Quality Issues
- Hardcoded file path ("test_pareto.csv")
- No algorithm-specific analysis
- Limited to single Pareto front
- Visualization missing axis labels in some plots

---

### 7. **comparing_results.ipynb** - Cross-Algorithm Comparison
**Type**: Jupyter Notebook  
**Purpose**: Compare optimization results across algorithms and classes

#### Analysis Structure
1. **Multi-Class Results Aggregation**:
   - Loads results from 6 object classes
   - Iterates: car → chair → person → pizza → TV → umbrella
2. **Pareto Front Extraction**:
   - Best individual per run: `argmax(fitness_yolo)`
3. **Metrics Extraction**:
   - CFG scale: `np.mean([...]) ± np.std([...])`
   - Guidance rescale statistics
   - Inference steps (f2) statistics
   - Convergence generation analysis
4. **Execution Time Calculation**:
   - Parses timestamps from runs_summary.csv
   - Calculates inter-run duration
   - Filters outliers (> 300 min assumed as restart)

#### Key Output
Comparison table with uncertainties:
```
Metric                    Our Results          StableYOLO Results
Guidance Scale (CFG)      X ± σ               [0, 20]
Guidance Rescale          X ± σ               -
Inference Steps           X ± σ               45.3 ± 30.29
Convergence Iterations    X ± σ               45.05 ± 10.01
Time per Run              X minutes           ~120 minutes
```

#### Issues
- Hardcoded class list
- Assumes consistent CSV structure across all classes
- No handling of missing Pareto files
- No inter-algorithm comparison (only within-class)

---

### 8. **robustness_analysis.ipynb** - Robustness & Stability Analysis
**Type**: Jupyter Notebook (~200+ cells)  
**Purpose**: Multi-faceted robustness evaluation

#### Part 1: Algorithm-Level Robustness (by run)
**Method**: Coefficient of Variation (CV = σ/μ)

**Classification Scheme**:
```
CV < 0.05        → High Robustness
0.05 ≤ CV ≤ 0.10 → Acceptable Robustness
CV > 0.10        → Low Robustness
```

**Metrics Evaluated**:
- hypervolume_feas (convergence quality)
- hypervolume_norm (normalized HV)
- f1_norm, f2_norm (normalized objectives)
- n_gen_real (actual generations)

**Output**: Boxplots per algorithm showing CV distribution

#### Part 2: Temporal Robustness (by generation)
**Method**: CV calculated across generations within each run

**Key Insight**: Tracks convergence stability
- Stable algorithm: HV increases monotonically, low CV
- Unstable algorithm: HV fluctuates, high CV

#### Part 3: Visual Analysis
**Visualization Techniques**:
1. Boxplots: CV distribution per algorithm
2. Seaborn violinplots: Probability density of HV
3. Stratified by run_id with monochromatic color scales:
   - NSGA2: Blue gradient
   - SMS-EMOA: Green gradient
   - MOEA/D: Red gradient

#### Issues
- Multiple redundant analyses (some overlapping)
- Hardcoded reference point for HV
- No statistical significance tests (p-values)
- Assumes uniform run_id structure (01..10)

---

### 9. **statistic_tests.ipynb** - Statistical Testing
**Type**: Jupyter Notebook (~126 lines, mostly skeleton)

#### Implemented Components
1. **Data Loading**: Multi-algorithm CSV concatenation
2. **Global Normalization**:
   - Min-max scaling: `(x - min) / (max - min)`
   - Handles edge case where max = min
   - Preserves order of algorithms: MOEAD → NSGA2 → SMS-EMOA

#### Missing Components
- Hypothesis testing (t-tests, Kruskal-Wallis)
- Effect size calculations (Cohen's d)
- Pairwise comparisons (Tukey HSD)
- Multi-objective ranking methods (Friedman test)

#### Code Structure Issues
- Notebook ends abruptly after normalization
- No statistical tests implemented
- Would benefit from moving to `statistic_tests.py` module

---

### 10. **requirements.txt** - Dependency Management
**Type**: Configuration file  
**Python Version**: 3.12  
**Total Packages**: 20+

#### Critical Version Pinning
```
torch==2.4.1              # Must match torchvision
torchvision==0.19.1       # CUDA acceleration
numpy>=1.26.4             # Python 3.12 minimum (< 1.26 incompatible)
scipy>=1.12.0             # Old scipy fails on Py3.12
pandas>=2.2.0             # Version 2.0.3 fails on Py3.12
pymoo==0.6.1.6            # Exact version (often has API changes)
ultralytics>=8.3.160      # YOLOv8 with pip fix
diffusers>=0.34.0         # Stable Diffusion
transformers>=4.46.3      # Tokenizers for SD
```

#### Known Issues
- Very strict version requirements (Python 3.12 compatibility)
- Potential conflicts with newer PyTorch releases
- HuggingFace token must be set in environment

---

## 🔄 ARCHITECTURE & WORKFLOW

### Optimization Pipeline

```
┌─────────────────────────────────────────────────────────────┐
│                    MAIN EXECUTION LOOP                       │
└─────────────────────────────────────────────────────────────┘
                              │
                ┌─────────────┼─────────────┐
                │             │             │
         ┌──────▼───────┐  ┌──▼──────────┐ ┌───▼──────────┐
         │    NSGA-II   │  │  MOEA/D    │ │ SMS-EMOA    │
         └──────┬───────┘  └──┬─────────┘ └───┬────────┘
                │             │              │
         ┌──────┴─────────────┴──────────────┴─────┐
         │  For each Operator Combination:        │
         │  - SBX × Polynomial Mutation           │
         │  - SBX × Gaussian Mutation             │
         │  - Uniform × Polynomial Mutation       │
         │  - Uniform × Gaussian Mutation         │
         └──────┬────────────────────────────────┘
                │
         ┌──────▼──────────────────────────┐
         │  For each Run (1..10)            │
         └──────┬──────────────────────────┘
                │
         ┌──────▼──────────────────────────┐
         │    Initialize Problem            │
         │  (StableDiffusionProblem)        │
         └──────┬──────────────────────────┘
                │
    ┌───────────▼────────────────┐
    │  Initialize Algorithm      │
    │  (with operators, params)  │
    └───────────┬────────────────┘
                │
    ┌───────────▼────────────────────┐
    │   Minimize (Optimization)       │
    │   Pop Size: 20-40              │
    │   Max Gens: 50-100             │
    │   Termination: FeasTol or Gen  │
    └───────────┬────────────────────┘
                │
    ┌───────────▼────────────────────────────┐
    │        Post-Optimization               │
    │  1. Filter feasible solutions          │
    │  2. Extract non-dominated front        │
    │  3. Calculate Hypervolume              │
    │  4. Log metrics to CSV                 │
    └────────────────────────────────────────┘
```

### Evaluation Pipeline (per solution)

```
┌─────────────────────────────────────────┐
│  Candidate Solution x = [iter, cfg,     │
│                          seed, rescale] │
└────────────┬────────────────────────────┘
             │
    ┌────────▼─────────────────────┐
    │  Generate Image               │
    │  (Stable Diffusion 2.1)       │
    │  - Seeds iter = x[0]          │
    │  - cfg = x[1]                 │
    │  - seed = x[2]                │
    │  - rescale = x[3]             │
    └────────┬──────────────────────┘
             │
    ┌────────▼──────────────────────┐
    │  Analyze Image                 │
    │  (YOLOv8 Detection)            │
    │  - Count objects               │
    │  - Extract confidences         │
    │  - Mean confidence = f₁        │
    └────────┬──────────────────────┘
             │
    ┌────────▼──────────────────────┐
    │  Compute Objectives            │
    │  f₁ = -mean_confidence         │
    │  f₂ = iterations (x[0])        │
    │  g₁ = 0.1 - mean_confidence   │
    └────────┬──────────────────────┘
             │
    ┌────────▼──────────────────────┐
    │  Return [f₁, f₂], [g₁]        │
    │  to Optimization Algorithm    │
    └───────────────────────────────┘
```

### Data Flow

```
problema.py
    ├─ Loads Global Models (GPU)
    │   ├─ StableDiffusionPipeline (∼4GB)
    │   └─ YOLOv8n (∼100MB)
    │
    └─ StableDiffusionProblem
        ├─ Input: x[0..3] (decision variables)
        ├─ Output: F[f1, f2], G[g1] (objectives & constraints)
        └─ Side-effect: Save images (optional)

main_notebook.ipynb
    ├─ Instantiates algorithm (NSGA2 / MOEAD / SMSEMOA)
    ├─ Calls minimize(problem, algorithm)
    ├─ Receives result object (res)
    │
    └─ Post-processes results:
        ├─ save_pareto()        → paretos/{name}.csv
        ├─ save_run_summary()   → runs_summary.csv
        └─ save_history_csv()   → history.csv

Analysis Notebooks
    ├─ analisis_pareto.ipynb    → Best solutions analysis
    ├─ comparing_results.ipynb  → Cross-class summary
    ├─ robustness_analysis.ipynb → Algorithm stability
    └─ statistic_tests.ipynb    → Statistical testing
```

---

## ⚠️ CURRENT ISSUES & CODE QUALITY

### Critical Issues

| Issue | File | Severity | Impact |
|-------|------|----------|--------|
| **Hardcoded Colab paths** | utils.py | CRITICAL | Code non-portable; fails on local machines |
| **CSV logic in notebook** | main_notebook.ipynb | HIGH | Hard to test, reuse, maintain |
| **Global model loading** | problema.py | HIGH | Difficult to parallelize; GPU memory issues |
| **No error recovery** | main_notebook.ipynb | HIGH | Failed runs not resumable |
| **Operator params not configurable** | operadores.py | MEDIUM | Can't tune without code changes |
| **Magic numbers** | main_notebook.ipynb | MEDIUM | Reference points, pop sizes, params scattered |

### Design Smell: Separation of Concerns

**notebook-level concerns (should be in modules)**:
```python
# In main_notebook.ipynb (SHOULD BE IN SEPARATE FILE)
_ensure_header()        # CSV utilities
_append_row()           # CSV utilities
save_pareto()           # Data persistence
save_run_summary()      # Data persistence
save_history_csv()      # Data persistence
setup_logger()          # Logging setup
get_termination_custom()# Optimization config
```

**Result**: 
- Notebooks mixed with production logic
- No unit tests possible
- Difficult to parallelize
- Hard to version control

### Code Organization Issues

| Module | Quality | Issues |
|--------|---------|--------|
| problema.py | Good | Global model state, could add docstrings |
| operadores.py | Fair | No validation, minimal docs |
| utils.py | Poor | Colab-specific, missing error handling |
| termination.py | Good | Minimal but functional |
| main_notebook.ipynb | Poor | CSV logic embedded, long cells, magic numbers |
| analisis_pareto.ipynb | Fair | Hardcoded paths, limited scope |
| robustness_analysis.ipynb | Fair | Some redundancy, good visualizations |
| statistic_tests.ipynb | Poor | Incomplete skeleton |

---

## 🔧 RECOMMENDED REFACTORING STRATEGY

### Phase 1: Extract Utilities (High Priority)

**Action 1.1: Create `data_io.py`**
Extract all CSV logic from notebook:
```python
# data_io.py
class CSVManager:
    """Manages CSV I/O for optimization results."""
    
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        self.pareto_dir = os.path.join(output_dir, "paretos")
        os.makedirs(self.pareto_dir, exist_ok=True)
    
    def ensure_header(self, filename: str, fieldnames: list) → None:
        """Create CSV with header if missing."""
    
    def append_row(self, filename: str, row: dict, fieldnames: list) → None:
        """Append row to CSV with field ordering."""
    
    def write_rows(self, filename: str, rows: list[dict], fieldnames: list) → None:
        """Write multiple rows to CSV."""
    
    def save_pareto_front(self, res, experiment_id: str, constraint_name: str = "g1") → str:
        """Save non-dominated feasible front."""
        # Extract feasible solutions
        # Apply NDS filter
        # Return path to CSV
    
    def save_run_metrics(self, algorithm: str, experiment_id: str, run_id: int, 
                        res, hv_value: float, operators: dict) → None:
        """Log per-run summary metrics."""
    
    def save_convergence_history(self, experiment_id: str, run_id: int, 
                                 res, ref_point: np.ndarray) → None:
        """Log generation-by-generation convergence."""
```

**Action 1.2: Create `config.py`**
Centralize all magic numbers and configuration:
```python
# config.py
from dataclasses import dataclass
from typing import List

@dataclass
class OptimizationConfig:
    """Global optimization configuration."""
    # Population parameters
    pop_size: int = 30
    n_max_gen: int = 100
    
    # Problem parameters
    ref_point_hv: List[float] = (0.0, 80)  # For HV calculation
    constraint_tolerance: float = 0.1       # fitness_yolo >= 0.1
    
    # Operator parameters
    sbx_prob: float = 0.9
    sbx_eta: float = 15
    polynomial_mutation_prob: float = 0.2
    polynomial_mutation_eta: float = 20
    gaussian_mutation_sigma: float = 0.1
    
    # Termination parameters
    ftol: float = 1e-4
    check_period: int = 5
    
    # Problem parameters
    iterations_range: tuple = (1, 100)
    cfg_range: tuple = (1, 20)
    seed_range: tuple = (0, 10000)
    guidance_rescale_range: tuple = (0, 1)
    
    # Data paths
    output_base_dir: str = "results"
```

**Action 1.3: Fix `utils.py`**
Remove Google Colab hardcoding:
```python
# utils.py (refactored)
import os
from pathlib import Path

def generar_imagen(pipe, prompt, negative_prompt, steps, cfg, seed, rescale):
    """Generate image via Stable Diffusion."""
    # UNCHANGED
    
def analizar_imagen_memoria(model, image):
    """Analyze image in memory (preferred)."""
    # UNCHANGED
    
def analizar_imagen(model, image_path):
    """Analyze image from file (fallback)."""
    # UNCHANGED
    
def prepare_output_directory(base_dir: str, experiment_id: str) -> Path:
    """Create output directory with proper structure."""
    out_dir = Path(base_dir) / experiment_id
    out_dir.mkdir(parents=True, exist_ok=True)
    return out_dir

def save_image_safely(image, output_path: str) -> bool:
    """Save image with error handling."""
    try:
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        image.save(output_path)
        return True
    except Exception as e:
        print(f"Failed to save image {output_path}: {e}")
        return False
    
# REMOVE Google Drive specific functions
# OR make them optional:
def archive_results(folder_path: str, output_dir: str = None) -> str:
    """Archive results (local-only version)."""
    if output_dir is None:
        output_dir = os.path.dirname(folder_path)
    # ... local zipfile logic
```

---

### Phase 2: Create Analysis Modules (Medium Priority)

**Action 2.1: Create `analysis.py`**
Move notebook functions to reusable module:
```python
# analysis.py
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, List

class ParetoAnalysis:
    """Analyze Pareto fronts from optimization results."""
    
    def __init__(self, pareto_csv_path: str):
        self.df = pd.read_csv(pareto_csv_path)
        self._validate()
    
    def _validate(self) → None:
        """Ensure required columns exist."""
        required = ["f1_alg", "f2"]
        missing = set(required) - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns: {missing}")
        self.df = self.df.dropna().reset_index(drop=True)
        self.df = self.df.drop_duplicates()
    
    def get_ideal_point(self) → Tuple[float, float]:
        """Get ideal point (minimum both objectives)."""
        f1_min = self.df["f1_alg"].min()
        f2_min = self.df["f2"].min()
        return (f1_min, f2_min)
    
    def get_best_solution(self, method: str = "euclidean") → pd.Series:
        """Find best solution by distance from ideal."""
        ideal = np.array(self.get_ideal_point())
        F = self.df[["f1_alg", "f2"]].values
        distances = np.linalg.norm(F - ideal, axis=1)
        return self.df.loc[distances.argmin()]
    
    def filter_by_constraint(self, column: str, threshold: float) → "ParetoAnalysis":
        """Filter solutions by constraint."""
        self.df = self.df[self.df[column] >= threshold]
        return self

class RobustnessAnalysis:
    """Analyze algorithm robustness across runs."""
    
    def __init__(self, history_csv_path: str):
        self.df = pd.read_csv(history_csv_path)
    
    def compute_cv_per_run(self, metric: str) → pd.DataFrame:
        """Coefficient of variation per run."""
        grouped = self.df.groupby("run_id")[metric]
        cv = (grouped.std() / grouped.mean()).rename(f"{metric}_cv")
        return cv.to_frame()
    
    def classify_robustness(self, cv_value: float) → str:
        """Classify robustness by CV threshold."""
        if pd.isna(cv_value) or np.isinf(cv_value):
            return "Undefined"
        if cv_value < 0.05:
            return "High"
        if cv_value <= 0.10:
            return "Acceptable"
        return "Low"

class ComparisonMetrics:
    """Compare results across algorithms."""
    
    @staticmethod
    def load_all_summaries(results_dir: str) -> pd.DataFrame:
        """Load and concatenate summaries from all algorithms."""
        dfs = []
        for alg_dir in Path(results_dir).iterdir():
            if alg_dir.is_dir():
                csv_path = alg_dir / "runs_summary.csv"
                if csv_path.exists():
                    df = pd.read_csv(csv_path)
                    df["algorithm"] = alg_dir.name
                    dfs.append(df)
        return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
    
    @staticmethod
    def compare_metrics(df: pd.DataFrame, groupby: str = "algorithm") → pd.DataFrame:
        """Compare key metrics across groups."""
        metrics = ["f1_min_feas", "f2_min_feas", "hypervolume_feas", "n_gen_real"]
        result = df.groupby(groupby)[metrics].agg(["mean", "std"])
        return result
```

**Action 2.2: Create `visualization.py`**
Centralize plotting code:
```python
# visualization.py
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

class ParetoPlotter:
    """Generate Pareto front visualizations."""
    
    def __init__(self, output_dir: str = "graphics"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_pareto_front(self, pareto_df, title: str, figsize=(10, 6)) → str:
        """Plot Pareto front scatter."""
        plt.figure(figsize=figsize)
        plt.scatter(pareto_df["f1_alg"], pareto_df["f2"], alpha=0.7)
        plt.xlabel("Objective 1 (Quality)")
        plt.ylabel("Objective 2 (Cost)")
        plt.title(title)
        plt.grid(True)
        out_path = self.output_dir / f"{title.replace(' ', '_')}.pdf"
        plt.savefig(out_path)
        plt.close()
        return str(out_path)
    
    def plot_convergence(self, history_df, algorithm: str) → str:
        """Plot HV convergence across generations."""
        # ... seaborn lineplot with confidence bands

class RobustnessPlotter:
    """Generate robustness analysis visualizations."""
    
    def __init__(self, output_dir: str = "graphics/robustness"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def plot_hv_by_algorithm(self, df, metric: str = "hypervolume_feas") → str:
        """Boxplot of HV distribution by algorithm."""
        plt.figure(figsize=(10, 6))
        sns.boxplot(data=df, x="algorithm", y=metric)
        plt.title(f"{metric} Distribution by Algorithm")
        out_path = self.output_dir / f"{metric}_by_algorithm.pdf"
        plt.savefig(out_path)
        plt.close()
        return str(out_path)
```

---

### Phase 3: Refactor Notebooks (High Priority)

**Action 3.1: Create `main_optimization.py`**
Convert main_notebook logic to executable script:
```python
# main_optimization.py
"""
Multi-objective optimization runner for Stable Diffusion parameters.

Usage:
    python main_optimization.py --algorithm nsga2 --runs 10 --output results/
"""

import argparse
import logging
from pathlib import Path
from datetime import datetime
import json

from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.algorithms.moo.sms import SMSEMOA
from pymoo.optimize import minimize
from pymoo.indicators.hv import Hypervolume

from problema import StableDiffusionProblem
from operadores import get_crossover, get_mutation
from config import OptimizationConfig
from data_io import CSVManager

def setup_logger(output_dir: str) -> logging.Logger:
    """Configure logging."""
    log_file = Path(output_dir) / "optimization.log"
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def get_algorithm(algo_name: str, cfg: OptimizationConfig, 
                  crossover: str, mutation: str):
    """Instantiate algorithm with operators."""
    
    cx = get_crossover(crossover)
    mut = get_mutation(mutation)
    
    if algo_name.upper() == "NSGA2":
        return NSGA2(pop_size=cfg.pop_size, sampling="random", 
                    crossover=cx, mutation=mut, eliminate_duplicates=True)
    elif algo_name.upper() == "MOEAD":
        ref_dirs = get_reference_directions("das-dennis", n_dim=2, n_partitions=10)
        return MOEAD(ref_dirs=ref_dirs, pop_size=cfg.pop_size,
                    crossover=cx, mutation=mut)
    elif algo_name.upper() == "SMSEMOA":
        return SMSEMOA(pop_size=cfg.pop_size, crossover=cx, mutation=mut)
    else:
        raise ValueError(f"Unknown algorithm: {algo_name}")

def run_optimization(algorithm_name: str, prompt: str, runs: int = 10,
                    output_dir: str = "results", save_images: bool = False):
    """Execute full optimization pipeline."""
    
    cfg = OptimizationConfig()
    logger = setup_logger(output_dir)
    csv_mgr = CSVManager(output_dir)
    
    logger.info(f"Starting optimization: {algorithm_name}")
    logger.info(f"Prompt: {prompt}")
    logger.info(f"Runs: {runs}")
    
    operators = [
        ("sbx", "polynomial"),
        ("sbx", "gaussian"),
        ("uniform", "polynomial"),
        ("uniform", "gaussian"),
    ]
    
    hv_calc = Hypervolume(ref_point=cfg.ref_point_hv)
    
    for op_idx, (cx_name, mut_name) in enumerate(operators):
        logger.info(f"Operator combo {op_idx+1}/4: {cx_name} + {mut_name}")
        
        for run_id in range(1, runs + 1):
            try:
                experiment_id = f"{algorithm_name}_{cx_name}_{mut_name}_run{run_id:02d}"
                logger.info(f"  Run {run_id}/{runs}: {experiment_id}")
                
                # Create problem
                problem = StableDiffusionProblem(
                    save_images=save_images,
                    prompt_parameter=prompt
                )
                
                # Get algorithm
                algorithm = get_algorithm(algorithm_name, cfg, cx_name, mut_name)
                
                # Optimize
                res = minimize(
                    problem,
                    algorithm,
                    ("n_gen", cfg.n_max_gen),
                    seed=run_id,
                    verbose=False
                )
                
                # Log results
                feasible_mask = np.all(np.asarray(res.pop.get("G")) <= 0, axis=1)
                F_feasible = np.asarray(res.pop.get("F"))[feasible_mask]
                hv = hv_calc.do(F_feasible) if len(F_feasible) > 0 else 0.0
                
                csv_mgr.save_pareto_front(res, experiment_id)
                csv_mgr.save_run_metrics(algorithm_name, experiment_id, run_id, res, hv,
                                         {"crossover": cx_name, "mutation": mut_name})
                csv_mgr.save_convergence_history(experiment_id, run_id, res, cfg.ref_point_hv)
                
                logger.info(f"    ✓ HV = {hv:.4f}")
                
            except Exception as e:
                logger.error(f"    ✗ Error in run {run_id}: {e}", exc_info=True)
                csv_mgr.save_error(experiment_id, run_id, algorithm_name, cx_name, mut_name, str(e))
                continue

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run multi-objective optimization")
    parser.add_argument("--algorithm", choices=["nsga2", "moead", "smsemoa"], required=True)
    parser.add_argument("--prompt", default="golden retriever dog")
    parser.add_argument("--runs", type=int, default=10)
    parser.add_argument("--output", default="results")
    parser.add_argument("--save-images", action="store_true")
    
    args = parser.parse_args()
    
    run_optimization(
        args.algorithm.upper(),
        args.prompt,
        runs=args.runs,
        output_dir=args.output,
        save_images=args.save_images
    )
```

**Action 3.2: Refactor Analysis Notebooks**
Convert to notebooks that call modules:
```python
# In comparing_results.ipynb (refactored)
from analysis import ComparisonMetrics
from visualization import RobustnessPlotter

# Instead of inline code:
metrics_df = ComparisonMetrics.load_all_summaries("results")
comparison = ComparisonMetrics.compare_metrics(metrics_df)

# Instead of inline plotting:
plotter = RobustnessPlotter()
plotter.plot_hv_by_algorithm(metrics_df, "hypervolume_feas")
```

---

### Phase 4: Create Testing Framework (Medium Priority)

**Action 4.1: Create `test_problem.py`**
```python
# test_problem.py
import unittest
import numpy as np
from problema import StableDiffusionProblem

class TestProblem(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        """Initialize problem once for all tests."""
        cls.problem = StableDiffusionProblem(save_images=False, prompt_parameter="dog")
    
    def test_problem_dimensions(self):
        """Verify problem dimensions."""
        self.assertEqual(self.problem.n_var, 4)
        self.assertEqual(self.problem.n_obj, 2)
        self.assertEqual(self.problem.n_constr, 1)
    
    def test_evaluation_shape(self):
        """Test evaluation returns correct shapes."""
        x = np.array([[50, 7.5, 42, 0.5]])  # Single solution
        out = {}
        self.problem._evaluate(x, out)
        
        self.assertEqual(out["F"].shape, (1, 2))
        self.assertEqual(out["G"].shape, (1, 1))
    
    def test_feasibility_constraint(self):
        """Test constraint feasibility."""
        # Should be feasible
        x = np.array([[50, 10, 42, 0.5]])
        out = {}
        self.problem._evaluate(x, out)
        # g1 = 0.1 - fitness_yolo
        # If fitness_yolo > 0.1, then g1 < 0 → feasible
    
    def test_variable_bounds(self):
        """Test variable bounds respected."""
        # Lower bounds
        x_lower = np.array([[1, 1, 0, 0]])
        # Upper bounds
        x_upper = np.array([[100, 20, 10000, 1]])
        
        # No exceptions should be raised
        out_l, out_u = {}, {}
        self.problem._evaluate(x_lower, out_l)
        self.problem._evaluate(x_upper, out_u)

if __name__ == "__main__":
    unittest.main()
```

---

### Phase 5: Documentation (Medium Priority)

**Action 5.1: Create `README.md`**
```markdown
# Multi-Objective Optimization for Stable Diffusion Image Generation

## Quick Start

### Installation
```bash
pip install -r requirements.txt
export HF_TOKEN="your_huggingface_token"
```

### Run Single Optimization
```bash
python main_optimization.py --algorithm nsga2 --runs 10 --output results/
```

### Run All Algorithms
```bash
for algo in nsga2 moead smsemoa; do
    python main_optimization.py --algorithm $algo --runs 10
done
```

### Analyze Results
```python
from analysis import ParetoAnalysis, RobustnessAnalysis
from visualization import ParetoPlotter

# Load Pareto front
pareto = ParetoAnalysis("results/nsga2/paretos/pareto_nsga2_run01.csv")
best = pareto.get_best_solution()
print(best)

# Plot
plotter = ParetoPlotter()
plotter.plot_pareto_front(pareto.df, "NSGA-II Results")
```

## Project Structure
- `problema.py`: MOO problem definition
- `operadores.py`: Genetic operators
- `config.py`: Centralized configuration
- `data_io.py`: CSV I/O management
- `analysis.py`: Analysis utilities
- `visualization.py`: Plotting utilities
- `main_optimization.py`: Optimization runner
- `*_notebook.ipynb`: Jupyter analysis notebooks

## Documentation
See [PROJECT_ANALYSIS.md](PROJECT_ANALYSIS.md) for comprehensive architecture details.
```

---

## 📊 SUMMARY TABLE: RECOMMENDED REFACTORING PRIORITY

| Phase | Action | Priority | Files | Impact |
|-------|--------|----------|-------|--------|
| 1 | Extract CSV logic to `data_io.py` | **HIGH** | main_notebook → data_io.py | Testability, reusability |
| 1 | Create `config.py` | **HIGH** | main_notebook → config.py | Maintainability, configurability |
| 1 | Fix `utils.py` portability | **HIGH** | utils.py | Local machine support |
| 2 | Create `analysis.py` | MEDIUM | analisis_pareto.ipynb → analysis.py | Reusability, modularity |
| 2 | Create `visualization.py` | MEDIUM | robustness_analysis.ipynb → visualization.py | Consistency, reusability |
| 3 | Convert to `main_optimization.py` | **HIGH** | main_notebook.ipynb | Automation, versioning |
| 4 | Create test suite | MEDIUM | test_*.py | Quality assurance |
| 5 | Write documentation | LOW | README.md, docstrings | Onboarding, maintenance |

---

## 🎓 THESIS DOCUMENT RECOMMENDATIONS

### For Your Thesis, Include:
1. **Project Architecture Diagram** (data flow, dependencies)
2. **Algorithm Comparison Table** (NSGA2 vs MOEA/D vs SMS-EMOA)
3. **Results Summary Statistics** (HV, convergence time, feasibility %)
4. **Robustness Analysis** (CV by algorithm, stability metrics)
5. **Pareto Front Visualizations** (per algorithm, normalized objectives)
6. **Parameter Sensitivity Analysis** (operator effects on convergence)

### Code References:
- Cite clase `StableDiffusionProblem` for problem formulation
- Reference `config.py` for experimental parameters
- Link to `main_optimization.py` for reproducibility

---

This analysis should provide a solid foundation for your thesis documentation and guide the refactoring process systematically. Let me know if you need deeper dives into specific sections!

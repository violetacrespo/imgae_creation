# -*- coding: utf-8 -*-
"""
Data I/O management for optimization results.

Handles all CSV operations: creating headers, appending rows, saving Pareto fronts,
and logging convergence history. Extracted from main_notebook.ipynb for reusability
and testability.
"""

import os
import csv
import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting


class CSVManager:
    """
    Manages CSV I/O for optimization results.
    
    Provides robust CSV operations with:
    - Automatic header creation
    - Field ordering consistency
    - Error handling
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize CSV manager.
        
        Args:
            output_dir: Root directory for all results
        """
        self.output_dir = Path(output_dir)
        self.pareto_dir = self.output_dir / "paretos"
        self.pareto_dir.mkdir(parents=True, exist_ok=True)
        
        # Standard CSV file paths
        self.runs_summary_path = self.output_dir / "runs_summary.csv"
        self.history_path = self.output_dir / "history.csv"
        self.errors_path = self.output_dir / "runs_errors.csv"
    
    def ensure_header(self, filename: str, fieldnames: List[str]) -> None:
        """
        Create CSV with header if it doesn't exist.
        
        Args:
            filename: CSV filename in output_dir
            fieldnames: List of column names
        """
        filepath = self.output_dir / filename
        if not filepath.exists():
            with open(filepath, mode="w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
    
    def append_row(self, filename: str, row: Dict, fieldnames: List[str]) -> None:
        """
        Append a single row to CSV with consistent field ordering.
        
        Args:
            filename: CSV filename in output_dir
            row: Dictionary with row data
            fieldnames: Ordered list of column names (ensures consistency)
        """
        filepath = self.output_dir / filename
        self.ensure_header(filename, fieldnames)
        
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            # Write only fields that exist in fieldnames (handles missing keys)
            writer.writerow({k: row.get(k, None) for k in fieldnames})
    
    def write_rows(self, filename: str, rows: List[Dict], fieldnames: List[str]) -> None:
        """
        Append multiple rows to CSV.
        
        Args:
            filename: CSV filename in output_dir
            rows: List of dictionaries with row data
            fieldnames: Ordered list of column names
        """
        filepath = self.output_dir / filename
        self.ensure_header(filename, fieldnames)
        
        with open(filepath, mode="a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            for row in rows:
                writer.writerow({k: row.get(k, None) for k in fieldnames})
    
    def save_pareto_front(self, res, experiment_id: str, 
                         constraint_col: str = "G",
                         constraint_threshold: float = 0.0) -> Optional[str]:
        """
        Save non-dominated feasible front from result object.
        
        Extracts feasible solutions (meeting constraint) and applies
        non-dominated sorting to extract Pareto front.
        
        Args:
            res: Result object from pymoo.optimize.minimize()
            experiment_id: Unique identifier for this run
            constraint_col: Name of constraint in result
            constraint_threshold: Constraint is feasible if <= this value
        
        Returns:
            Path to saved CSV, or None if no feasible solutions
        """
        pop = getattr(res, "pop", None)
        if pop is None:
            return None
        
        # Extract decision variables and objectives
        X = pop.get("X")
        F = pop.get("F")
        G = pop.get("G")
        
        if X is None or F is None:
            return None
        
        X = np.asarray(X)
        F = np.asarray(F)
        
        # Filter by feasibility constraint
        if G is not None:
            G = np.asarray(G)
            # Feasible if all constraints <= 0
            feasible = np.all(G <= constraint_threshold, axis=1)
            X = X[feasible]
            F = F[feasible]
        
        # Check if any feasible solutions exist
        if F.shape[0] == 0:
            print(f"[save_pareto] No feasible solutions in {experiment_id}")
            return None
        
        # Extract non-dominated front
        nd_idx = NonDominatedSorting().do(F, only_non_dominated_front=True)
        X_nd = X[nd_idx]
        F_nd = F[nd_idx]
        
        # Create DataFrame
        columns_X = ["iterations", "cfg", "sd_seed", "guidance_rescale"]
        columns_F = ["f1_alg", "f2"]
        
        pareto_df = pd.DataFrame(
            np.hstack((X_nd, F_nd)),
            columns=columns_X + columns_F
        )
        
        # Add fitness (negate f1 to get original fitness_yolo)
        pareto_df["fitness_yolo"] = -pareto_df["f1_alg"]
        
        # Save to CSV
        out_csv = self.pareto_dir / f"pareto_{experiment_id}.csv"
        pareto_df.to_csv(out_csv, index=False)
        
        print(f"[save_pareto] Saved {len(nd_idx)} solutions to {out_csv}")
        return str(out_csv)
    
    def save_run_summary(self, algorithm_name: str, experiment_id: str, run_id: int,
                        res, hv_value: float, operators: Dict[str, str],
                        ref_point_hv: List[float]) -> None:
        """
        Save aggregated metrics for a single run.
        
        Args:
            algorithm_name: Name of algorithm (NSGA2, MOEAD, SMSEMOA)
            experiment_id: Unique identifier
            run_id: Run number (1-based)
            res: Result object from minimize()
            hv_value: Hypervolume of final solution set
            operators: Dict with 'crossover' and 'mutation' keys
            ref_point_hv: Reference point used for HV calculation
        """
        pop_final = res.pop
        F_final = np.asarray(pop_final.get("F"))
        G_final = pop_final.get("G")
        
        n_final_pop = int(F_final.shape[0])
        
        # Filter feasible solutions
        if G_final is not None:
            G_final = np.asarray(G_final)
            feas = np.all(G_final <= 0, axis=1)
            F_use = F_final[feas]
        else:
            F_use = F_final
        
        n_feasible_final = int(F_use.shape[0])
        
        # Compute statistics (handle empty case)
        if n_feasible_final > 0:
            f1_min = float(np.min(F_use[:, 0]))
            f1_mean = float(np.mean(F_use[:, 0]))
            f2_min = float(np.min(F_use[:, 1]))
            f2_mean = float(np.mean(F_use[:, 1]))
            fitness_yolo_max = float(np.max(-F_use[:, 0]))
            fitness_yolo_mean = float(np.mean(-F_use[:, 0]))
        else:
            f1_min = f1_mean = f2_min = f2_mean = None
            fitness_yolo_max = fitness_yolo_mean = None
        
        n_gen_real = getattr(res.algorithm, "n_gen", None)
        n_gen_real = int(n_gen_real) if n_gen_real is not None else None
        
        row = {
            "experiment_id": experiment_id,
            "algorithm": algorithm_name,
            "crossover": operators.get("crossover"),
            "mutation": operators.get("mutation"),
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            
            "pop_size": pop_final.__len__(),
            "n_gen_max": None,  # Should be passed as parameter if needed
            "n_gen_real": n_gen_real,
            
            "n_final_pop": n_final_pop,
            "n_feasible_final": n_feasible_final,
            
            "f1_min_feas": f1_min,
            "f1_mean_feas": f1_mean,
            "f2_min_feas": f2_min,
            "f2_mean_feas": f2_mean,
            
            "fitness_yolo_max_feas": fitness_yolo_max,
            "fitness_yolo_mean_feas": fitness_yolo_mean,
            
            "ref_point_hv": json.dumps(ref_point_hv),
            "hypervolume_feas": hv_value,
        }
        
        fieldnames = [
            "experiment_id", "algorithm", "crossover", "mutation", "run_id", "timestamp",
            "pop_size", "n_gen_max", "n_gen_real",
            "n_final_pop", "n_feasible_final",
            "f1_min_feas", "f1_mean_feas", "f2_min_feas", "f2_mean_feas",
            "fitness_yolo_max_feas", "fitness_yolo_mean_feas",
            "ref_point_hv", "hypervolume_feas"
        ]
        
        self.append_row("runs_summary.csv", row, fieldnames)
    
    def save_convergence_history(self, experiment_id: str, run_id: int, res,
                                ref_point_hv: List[float]) -> None:
        """
        Save generation-by-generation convergence history.
        
        Args:
            experiment_id: Unique identifier
            run_id: Run number
            res: Result object from minimize()
            ref_point_hv: Reference point for HV calculation
        """
        if not hasattr(res, "history") or res.history is None:
            return
        
        from pymoo.indicators.hv import Hypervolume
        
        rows = []
        hv_ref = np.array(ref_point_hv, dtype=float)
        
        for gen_idx, hist in enumerate(res.history):
            pop = hist.pop
            Fg = np.asarray(pop.get("F"))
            Gg = pop.get("G")
            n_pop = int(Fg.shape[0])
            
            # Filter feasible
            if Gg is not None:
                Gg = np.asarray(Gg)
                feas = np.all(Gg <= 0, axis=1)
                F_use = Fg[feas]
            else:
                feas = np.ones(n_pop, dtype=bool)
                F_use = Fg
            
            n_feas = int(np.sum(feas))
            
            # Calculate HV
            hv_calc = Hypervolume(ref_point=hv_ref)
            hv_val = float(hv_calc.do(F_use)) if F_use.shape[0] > 0 else None
            
            # Compute statistics
            if F_use.shape[0] > 0:
                f1_min = float(np.min(F_use[:, 0]))
                f2_min = float(np.min(F_use[:, 1]))
                f1_mean = float(np.mean(F_use[:, 0]))
                f2_mean = float(np.mean(F_use[:, 1]))
                fitness_yolo_max = float(np.max(-F_use[:, 0]))
                fitness_yolo_mean = float(np.mean(-F_use[:, 0]))
            else:
                f1_min = f2_min = f1_mean = f2_mean = None
                fitness_yolo_max = fitness_yolo_mean = None
            
            rows.append({
                "experiment_id": experiment_id,
                "run_id": run_id,
                "gen": int(gen_idx),
                "n_pop": n_pop,
                "n_feasible": n_feas,
                "f1_min_feas": f1_min,
                "f2_min_feas": f2_min,
                "f1_mean_feas": f1_mean,
                "f2_mean_feas": f2_mean,
                "fitness_yolo_max_feas": fitness_yolo_max,
                "fitness_yolo_mean_feas": fitness_yolo_mean,
                "ref_point_hv": json.dumps(ref_point_hv),
                "hypervolume_feas": hv_val,
            })
        
        fieldnames = [
            "experiment_id", "run_id", "gen",
            "n_pop", "n_feasible",
            "f1_min_feas", "f2_min_feas", "f1_mean_feas", "f2_mean_feas",
            "fitness_yolo_max_feas", "fitness_yolo_mean_feas",
            "ref_point_hv", "hypervolume_feas"
        ]
        
        self.write_rows("history.csv", rows, fieldnames)
    
    def save_error(self, experiment_id: str, run_id: int, algorithm_name: str,
                  crossover: str, mutation: str, error_message: str) -> None:
        """
        Log error from failed run.
        
        Args:
            experiment_id: Unique identifier
            run_id: Run number
            algorithm_name: Algorithm name
            crossover: Crossover operator name
            mutation: Mutation operator name
            error_message: Error description
        """
        row = {
            "experiment_id": experiment_id,
            "algorithm": algorithm_name,
            "crossover": crossover,
            "mutation": mutation,
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "error": error_message,
        }
        
        fieldnames = [
            "experiment_id", "algorithm", "crossover", "mutation", "run_id",
            "timestamp", "error"
        ]
        
        self.append_row("runs_errors.csv", row, fieldnames)

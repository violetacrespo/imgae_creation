# -*- coding: utf-8 -*-
"""
Analysis utilities for optimization results.

Provides reusable analysis classes for Pareto front analysis, robustness
evaluation, and cross-algorithm comparison.
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Dict
from dataclasses import dataclass


@dataclass
class MetricsResult:
    """Container for aggregated metrics."""
    mean_values: Dict[str, float]
    std_values: Dict[str, float]
    cv_values: Dict[str, float]  # Coefficient of variation


class ParetoAnalysis:
    """
    Analyze Pareto fronts from optimization results.
    
    Provides methods for:
    - Loading and validating Pareto front data
    - Finding ideal and best solutions
    - Statistical analysis of front characteristics
    """
    
    def __init__(self, pareto_csv_path: str):
        """
        Initialize Pareto analysis from CSV file.
        
        Args:
            pareto_csv_path: Path to Pareto front CSV
        
        Raises:
            FileNotFoundError: If CSV doesn't exist
            ValueError: If required columns are missing
        """
        csv_path = Path(pareto_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"Pareto CSV not found: {pareto_csv_path}")
        
        self.df = pd.read_csv(csv_path)
        self._validate()
    
    def _validate(self) -> None:
        """Validate required columns and clean data."""
        required = ["f1_alg", "f2"]
        missing = set(required) - set(self.df.columns)
        
        if missing:
            raise ValueError(
                f"Missing required columns: {missing}. "
                f"Available columns: {list(self.df.columns)}"
            )
        
        # Clean: remove NaN and duplicates
        initial_size = len(self.df)
        self.df = self.df.dropna(subset=["f1_alg", "f2"]).reset_index(drop=True)
        self.df = self.df.drop_duplicates(subset=["f1_alg", "f2"]).reset_index(drop=True)
        final_size = len(self.df)
        
        if final_size < initial_size:
            print(f"[ParetoAnalysis] Cleaned data: {initial_size} → {final_size} rows")
    
    def get_ideal_point(self) -> Tuple[float, float]:
        """
        Get ideal point (minimum both objectives).
        
        Returns:
            Tuple of (min_f1, min_f2)
        """
        f1_min = self.df["f1_alg"].min()
        f2_min = self.df["f2"].min()
        return (f1_min, f2_min)
    
    def get_anti_ideal_point(self) -> Tuple[float, float]:
        """
        Get anti-ideal point (maximum both objectives).
        
        Returns:
            Tuple of (max_f1, max_f2)
        """
        f1_max = self.df["f1_alg"].max()
        f2_max = self.df["f2"].max()
        return (f1_max, f2_max)
    
    def get_best_solution(self, method: str = "euclidean") -> pd.Series:
        """
        Find best solution by distance from ideal point.
        
        Args:
            method: Distance metric ('euclidean', 'manhattan', 'chebyshev')
        
        Returns:
            Pandas Series representing best solution
        """
        ideal = np.array(self.get_ideal_point())
        F = self.df[["f1_alg", "f2"]].values
        
        if method == "euclidean":
            distances = np.linalg.norm(F - ideal, axis=1)
        elif method == "manhattan":
            distances = np.sum(np.abs(F - ideal), axis=1)
        elif method == "chebyshev":
            distances = np.max(np.abs(F - ideal), axis=1)
        else:
            raise ValueError(f"Unknown method: {method}")
        
        best_idx = np.argmin(distances)
        return self.df.loc[best_idx]
    
    def filter_by_constraint(self, column: str, threshold: float,
                           operation: str = ">=") -> "ParetoAnalysis":
        """
        Filter solutions by constraint.
        
        Args:
            column: Column name to filter
            threshold: Threshold value
            operation: Comparison operator ('>=', '<=', '>', '<', '==')
        
        Returns:
            New ParetoAnalysis object with filtered data
        """
        if operation == ">=":
            filtered = self.df[self.df[column] >= threshold]
        elif operation == "<=":
            filtered = self.df[self.df[column] <= threshold]
        elif operation == ">":
            filtered = self.df[self.df[column] > threshold]
        elif operation == "<":
            filtered = self.df[self.df[column] < threshold]
        elif operation == "==":
            filtered = self.df[self.df[column] == threshold]
        else:
            raise ValueError(f"Unknown operation: {operation}")
        
        result = ParetoAnalysis.__new__(ParetoAnalysis)
        result.df = filtered.reset_index(drop=True)
        return result
    
    def get_statistics(self) -> MetricsResult:
        """
        Compute statistics of Pareto front.
        
        Returns:
            MetricsResult with mean, std, and CV for f1 and f2
        """
        metrics = ["f1_alg", "f2"]
        means = {m: float(self.df[m].mean()) for m in metrics}
        stds = {m: float(self.df[m].std()) for m in metrics}
        cvs = {m: stds[m] / means[m] if means[m] != 0 else np.nan
               for m in metrics}
        
        return MetricsResult(means, stds, cvs)
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __repr__(self) -> str:
        return f"ParetoAnalysis(n_solutions={len(self)}, columns={list(self.df.columns)})"


class RobustnessAnalysis:
    """
    Analyze algorithm robustness across multiple runs.
    
    Provides methods for:
    - Computing coefficient of variation (CV)
    - Classifying robustness levels
    - Temporal stability analysis
    """
    
    def __init__(self, history_csv_path: str):
        """
        Initialize robustness analysis from history CSV.
        
        Args:
            history_csv_path: Path to history.csv file
        """
        csv_path = Path(history_csv_path)
        if not csv_path.exists():
            raise FileNotFoundError(f"History CSV not found: {history_csv_path}")
        
        self.df = pd.read_csv(csv_path)
    
    def compute_cv_per_run(self, metric: str) -> pd.DataFrame:
        """
        Compute coefficient of variation for each run.
        
        Args:
            metric: Metric column name
        
        Returns:
            DataFrame with CV values per run
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric {metric} not found in data")
        
        grouped = self.df.groupby("run_id")[metric]
        means = grouped.mean()
        stds = grouped.std()
        cvs = stds / means
        
        result = pd.DataFrame({
            f"{metric}_mean": means,
            f"{metric}_std": stds,
            f"{metric}_cv": cvs
        })
        
        return result
    
    def compute_cv_per_generation(self, metric: str) -> pd.DataFrame:
        """
        Compute CV across runs at each generation.
        
        Args:
            metric: Metric column name
        
        Returns:
            DataFrame with CV values per generation
        """
        if metric not in self.df.columns:
            raise ValueError(f"Metric {metric} not found in data")
        
        grouped = self.df.groupby("gen")[metric]
        means = grouped.mean()
        stds = grouped.std()
        cvs = stds / means
        
        result = pd.DataFrame({
            f"{metric}_mean": means,
            f"{metric}_std": stds,
            f"{metric}_cv": cvs
        })
        
        return result
    
    @staticmethod
    def classify_robustness(cv_value: float) -> str:
        """
        Classify robustness level based on CV threshold.
        
        Classification scheme:
        - CV < 0.05: High robustness
        - 0.05 <= CV <= 0.10: Acceptable robustness
        - CV > 0.10: Low robustness
        - CV is NaN or inf: Undefined
        
        Args:
            cv_value: Coefficient of variation
        
        Returns:
            Robustness classification string
        """
        if pd.isna(cv_value) or np.isinf(cv_value):
            return "Undefined"
        elif cv_value < 0.05:
            return "High Robustness"
        elif cv_value <= 0.10:
            return "Acceptable Robustness"
        else:
            return "Low Robustness"
    
    def classify_all_runs(self, metric: str) -> pd.DataFrame:
        """
        Classify robustness for all runs.
        
        Args:
            metric: Metric column name
        
        Returns:
            DataFrame with robustness classification per run
        """
        cv_df = self.compute_cv_per_run(metric)
        cv_df[f"{metric}_robustness"] = cv_df[f"{metric}_cv"].apply(
            self.classify_robustness
        )
        return cv_df


class ComparisonMetrics:
    """
    Compare results across multiple algorithms.
    
    Provides methods for:
    - Loading results from multiple algorithm runs
    - Normalizing metrics
    - Computing comparative statistics
    """
    
    @staticmethod
    def load_all_summaries(results_dir: str) -> pd.DataFrame:
        """
        Load and concatenate runs_summary.csv from all algorithms.
        
        Args:
            results_dir: Root results directory
        
        Returns:
            Concatenated DataFrame with algorithm column
        """
        results_path = Path(results_dir)
        dfs = []
        
        for alg_dir in results_path.iterdir():
            if not alg_dir.is_dir():
                continue
            
            csv_path = alg_dir / "runs_summary.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["algorithm"] = alg_dir.name
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    @staticmethod
    def load_all_history(results_dir: str) -> pd.DataFrame:
        """
        Load and concatenate history.csv from all algorithms.
        
        Args:
            results_dir: Root results directory
        
        Returns:
            Concatenated DataFrame with algorithm column
        """
        results_path = Path(results_dir)
        dfs = []
        
        for alg_dir in results_path.iterdir():
            if not alg_dir.is_dir():
                continue
            
            csv_path = alg_dir / "history.csv"
            if csv_path.exists():
                df = pd.read_csv(csv_path)
                df["algorithm"] = alg_dir.name
                dfs.append(df)
        
        if not dfs:
            return pd.DataFrame()
        
        return pd.concat(dfs, ignore_index=True)
    
    @staticmethod
    def normalize_metrics(df: pd.DataFrame, columns: list) -> pd.DataFrame:
        """
        Min-max normalize specified columns.
        
        Handles edge case where max == min (returns 0.0).
        
        Args:
            df: Input DataFrame
            columns: Column names to normalize
        
        Returns:
            DataFrame with normalized columns (suffixed with "_norm")
        """
        result = df.copy()
        
        for col in columns:
            if col not in result.columns:
                continue
            
            col_min = result[col].min()
            col_max = result[col].max()
            col_range = col_max - col_min
            
            if col_range == 0:
                result[f"{col}_norm"] = 0.0
            else:
                result[f"{col}_norm"] = (result[col] - col_min) / col_range
        
        return result
    
    @staticmethod
    def compare_metrics(df: pd.DataFrame, metrics: list,
                       groupby: str = "algorithm") -> pd.DataFrame:
        """
        Compare metrics across groups.
        
        Args:
            df: Input DataFrame
            metrics: Metric column names
            groupby: Column to group by (typically "algorithm")
        
        Returns:
            DataFrame with mean and std for each metric per group
        """
        return df.groupby(groupby)[metrics].agg(["mean", "std"])

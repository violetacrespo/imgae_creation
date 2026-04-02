# -*- coding: utf-8 -*-
"""
Global configuration for multi-objective optimization experiments.

This module centralizes all configuration parameters, eliminating magic numbers
scattered throughout notebooks and scripts.
"""

from dataclasses import dataclass
from typing import List, Tuple
import json


@dataclass
class OptimizationConfig:
    """
    Centralized configuration for optimization experiments.
    
    Attributes:
        pop_size: Population size for evolutionary algorithm
        n_max_gen: Maximum number of generations
        ref_point_hv: Reference point for hypervolume calculation
        constraint_threshold: Minimum fitness_yolo for feasibility (g1 <= 0)
        
        Operator parameters:
        sbx_prob: SBX crossover probability
        sbx_eta: SBX distribution index
        polynomial_mutation_prob: Polynomial mutation probability
        polynomial_mutation_eta: Polynomial mutation distribution index
        gaussian_mutation_sigma: Gaussian mutation standard deviation
        
        Problem parameters:
        iterations_range: Min/max inference steps for Stable Diffusion
        cfg_range: Min/max guidance scale
        seed_range: Min/max random seed
        guidance_rescale_range: Min/max guidance rescale factor
        
        Termination parameters:
        ftol: Function space tolerance for convergence
        check_period: Generations between convergence checks
        
        I/O parameters:
        output_base_dir: Root directory for results
        save_images: Whether to save generated images to disk
    """
    
    # ==================== POPULATION & GENERATIONS ====================
    pop_size: int = 30
    n_max_gen: int = 100
    
    # ==================== HYPERVOLUME ====================
    # Reference point for HV calculation
    # [0.0, 80]: Assumes f1 in [0,1] (neg fitness), f2 up to ~80 steps
    ref_point_hv: List[float] = None
    
    # ==================== CONSTRAINTS ====================
    # Constraint: g1 = constraint_threshold - fitness_yolo <= 0
    # Feasible if: fitness_yolo >= constraint_threshold
    constraint_threshold: float = 0.1
    
    # ==================== CROSSOVER OPERATORS ====================
    sbx_prob: float = 0.9          # Probability of SBX crossover
    sbx_eta: float = 15            # Distribution index (higher = less spread)
    
    # ==================== MUTATION OPERATORS ====================
    polynomial_mutation_prob: float = 0.2
    polynomial_mutation_eta: float = 20
    gaussian_mutation_sigma: float = 0.1
    
    # ==================== PROBLEM VARIABLE RANGES ====================
    iterations_range: Tuple[int, int] = (1, 100)
    cfg_range: Tuple[float, float] = (1, 20)
    seed_range: Tuple[int, int] = (0, 10000)
    guidance_rescale_range: Tuple[float, float] = (0, 1)
    
    # ==================== TERMINATION ====================
    ftol: float = 1e-4             # Function space tolerance
    xtol: float = 1e-8             # Design space tolerance
    cvtol: float = 1e-6            # Constraint violation tolerance
    check_period: int = 5          # Check convergence every N generations
    n_max_evals: int = 100000      # Max function evaluations
    
    # ==================== I/O ====================
    output_base_dir: str = "results"
    save_images: bool = False
    
    def __post_init__(self):
        """Validate and set defaults."""
        if self.ref_point_hv is None:
            self.ref_point_hv = [0.0, 80]
    
    def to_dict(self) -> dict:
        """Convert config to dictionary for JSON serialization."""
        config_dict = {}
        for key, value in self.__dict__.items():
            if isinstance(value, list):
                config_dict[key] = value
            elif isinstance(value, tuple):
                config_dict[key] = list(value)
            else:
                config_dict[key] = value
        return config_dict
    
    def to_json(self) -> str:
        """Serialize to JSON string."""
        return json.dumps(self.to_dict(), indent=2)
    
    @classmethod
    def from_dict(cls, config_dict: dict) -> "OptimizationConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in config_dict.items() if k in cls.__dataclass_fields__})


# Default configuration instance
DEFAULT_CONFIG = OptimizationConfig()


# Presets for different scenarios
class ConfigPresets:
    """Pre-defined configurations for different experiment types."""
    
    @staticmethod
    def fast_test() -> OptimizationConfig:
        """Quick test run: small populations, few generations."""
        cfg = OptimizationConfig()
        cfg.pop_size = 10
        cfg.n_max_gen = 10
        cfg.save_images = False
        return cfg
    
    @staticmethod
    def medium() -> OptimizationConfig:
        """Medium-length run: balanced parameters."""
        cfg = OptimizationConfig()
        cfg.pop_size = 30
        cfg.n_max_gen = 50
        cfg.save_images = False
        return cfg
    
    @staticmethod
    def long_run() -> OptimizationConfig:
        """Long run: larger populations, more generations."""
        cfg = OptimizationConfig()
        cfg.pop_size = 50
        cfg.n_max_gen = 100
        cfg.save_images = True
        return cfg
    
    @staticmethod
    def benchmark() -> OptimizationConfig:
        """Benchmark configuration: standardized parameters for comparison."""
        cfg = OptimizationConfig()
        cfg.pop_size = 30
        cfg.n_max_gen = 100
        cfg.ftol = 1e-4
        cfg.check_period = 5
        cfg.save_images = False
        return cfg

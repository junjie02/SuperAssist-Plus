"""Configuration for trajectory dataset generation."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class TrajectoryDatasetConfig:
    """Configuration for trajectory dataset generation.

    All fields can be overridden via CLI arguments.
    """

    # --- LLM configuration ---
    model: str = "deepseek:deepseek-v4-pro"
    temperature: float = 0.0
    max_tokens: int = 8192
    max_exploration_steps: int = 1

    # --- Output ---
    output_dir: str = "dataset/output"
    output_file: str = "musique_trajectories.jsonl"
    flush_every: int = 10

    # --- Sampling ---
    max_examples: int = 0          # 0 = all examples in the dataset
    max_trajectories: int = 10000  # 0 = no limit
    start_example_index: int = 0   # For resume: skip first N examples

    # --- Benchmark config ---
    benchmark_profile: str = "configs/musique_profile.yaml"
    data_path: str = ""            # Override data path (empty = default)
    limit_dataset: str = "answerable"  # "answerable", "all", or "first-N"

    # --- Graph context size ---
    context_max_nodes: int = 60
    context_max_chars: int = 20000

    # --- Quality filters ---
    include_parse_errors: bool = True
    min_operations: int = 0        # Minimum operations in UpdatePlan to include

    # --- Resume ---
    resume: bool = False

    @property
    def effective_output_path(self) -> Path:
        return Path(self.output_dir) / self.output_file

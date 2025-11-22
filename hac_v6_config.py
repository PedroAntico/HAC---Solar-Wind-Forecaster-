#!/usr/bin/env python3
"""
hac_v6_config.py
Configuration manager for HAC v6 Solar Wind Forecaster
"""

import os
import yaml

class HACConfig:
    """
    Loads and validates configuration from config.yaml
    Provides easy access via: config.get("section", {}).get("param")
    """

    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)

        self._validate_structure()

    # ------------------------------------------------------------
    # Access helper
    # ------------------------------------------------------------
    def get(self, key: str, default=None):
        return self.config_dict.get(key, default)

    # ------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------
    def _validate_structure(self):
        required_sections = [
            "paths",
            "training",
            "models",
            "horizons",
            "targets",
            "realtime"
        ]

        for section in required_sections:
            if section not in self.config_dict:
                raise ValueError(f"Missing required section in config.yaml: {section}")

        # Validate directory structure
        paths = self.config_dict["paths"]
        for key in ["data_dir", "model_dir", "results_dir", "logs_dir"]:
            if key in paths:
                os.makedirs(paths[key], exist_ok=True)

        # Validate horizons list
        horizons = self.config_dict.get("horizons", [])
        if not isinstance(horizons, list) or any(h < 1 for h in horizons):
            raise ValueError("Horizons must be a list of positive integers")

        # Validate targets
        if "primary" not in self.config_dict["targets"]:
            raise ValueError("You must define primary targets in config.yaml")

    # ------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------
    def data_files(self):
        """Return list of possible input data files"""
        base = self.get("paths", {}).get("data_dir", "data_real")
        files = self.get("paths", {}).get("data_files", [])
        return [os.path.join(base, f) for f in files]

    def model_dir(self):
        return self.get("paths", {}).get("model_dir")

    def results_dir(self):
        return self.get("paths", {}).get("results_dir")

    def logs_dir(self):
        return self.get("paths", {}).get("logs_dir")

    def __repr__(self):
        return f"HACConfig({self.config_dict})"

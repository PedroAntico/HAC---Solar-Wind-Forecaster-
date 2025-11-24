#!/usr/bin/env python3
"""
hac_v6_config.py
Config manager simples para o HAC v6
"""

import os
import yaml


class HACConfig:
    """
    Carrega e valida config.yaml
    Acesso via: config.get("paths")["data_dir"], etc.
    """

    def __init__(self, config_path: str = "config.yaml"):
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")

        with open(config_path, "r") as f:
            self.config_dict = yaml.safe_load(f)

        if not isinstance(self.config_dict, dict):
            raise ValueError("config.yaml inválido (não é um dicionário)")

        self._validate_structure()

    # ------------------------------------------------------------
    def get(self, key: str, default=None):
        """Acesso direto ao dicionário interno."""
        return self.config_dict.get(key, default)

    # ------------------------------------------------------------
    def _validate_structure(self):
        required_sections = [
            "paths",
            "training",
            "horizons",
            "targets"
        ]

        for section in required_sections:
            if section not in self.config_dict:
                raise ValueError(f"Missing required section in config.yaml: {section}")

        paths = self.config_dict["paths"]
        for key in ["data_dir", "model_dir", "results_dir", "logs_dir"]:
            if key in paths:
                os.makedirs(paths[key], exist_ok=True)

        horizons = self.config_dict.get("horizons", [])
        if not isinstance(horizons, list) or any(h < 1 for h in horizons):
            raise ValueError("horizons deve ser uma lista de inteiros positivos")

        if "primary" not in self.config_dict["targets"]:
            raise ValueError("Defina 'targets.primary' no config.yaml")

    # ------------------------------------------------------------
    def __repr__(self):
        return f"HACConfig({self.config_dict})"

# src/utility/config_loader.py
import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """YAML configuration loader for config/ folder in project root"""

    def __init__(self):
        # Always use config/ in project root
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent  # Go up to project root
        self.config_base_path = project_root / "config"

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load configuration from YAML file in config/ folder

        Args:
            config_name: Name of the config file (without .yaml extension)

        Returns:
            Configuration dictionary
        """
        config_file_path = self.config_base_path / f"{config_name}.yaml"

        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        return config

    def get(self, config_name: str, key: str, default: Any = None) -> Any:
        """
        Get specific configuration value using dot notation

        Args:
            config_name: Name of the config file
            key: Dot notation key (e.g., 'database.host')
            default: Default value if key not found

        Returns:
            Configuration value
        """
        config = self.load_config(config_name)

        # Navigate using dot notation
        keys = key.split(".")
        current = config

        for k in keys:
            if isinstance(current, dict) and k in current:
                current = current[k]
            else:
                return default

        return current


# Singleton instance
config_loader = ConfigLoader()

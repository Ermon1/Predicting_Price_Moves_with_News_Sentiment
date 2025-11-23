import yaml
from pathlib import Path
from typing import Any, Dict


class ConfigLoader:
    """Industry-grade configuration loader - DO NOT MODIFY IN TASKS"""

    def __init__(self):
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent.parent
        self.config_base_path = project_root / "config"

    def load_config(self, config_name: str) -> Dict[str, Any]:
        """
        Load entire configuration from YAML file

        Args:
            config_name: Name of config file without .yaml extension

        Returns:
            Dictionary with all configuration data
        """
        config_file_path = self.config_base_path / f"{config_name}.yaml"

        if not config_file_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_file_path}")

        with open(config_file_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file) or {}

        return config

    def get_task_config(self, config_name: str, task: str) -> Dict[str, Any]:
        """
        Get merged configuration for specific task
        Returns: base_processing + task_specific[task] merged together

        Args:
            config_name: Name of config file (e.g., 'processing_pipeline')
            task: Task name (e.g., 'task1', 'task2', 'task3')

        Returns:
            Dictionary with merged configuration for the task
        """
        config = self.load_config(config_name)

        # Get base settings that all tasks share
        base_settings = config.get("base_processing", {})

        # Get task-specific settings that override base settings
        task_specific = config.get("task_specific", {}).get(task, {})

        # Merge them (task-specific settings override base settings)
        merged_config = self._deep_merge(base_settings, task_specific)

        print(f"✓ Loaded configuration for: {task}")
        return merged_config

    def _deep_merge(self, base: Dict, override: Dict) -> Dict:
        """
        Deep merge two dictionaries - override values take priority

        Args:
            base: Base dictionary with default settings
            override: Override dictionary with task-specific settings

        Returns:
            Merged dictionary
        """
        result = base.copy()
        for key, value in override.items():
            if (
                key in result
                and isinstance(result[key], dict)
                and isinstance(value, dict)
            ):
                # Recursively merge nested dictionaries
                result[key] = self._deep_merge(result[key], value)
            else:
                # Override the value
                result[key] = value
        return result


# Singleton instance - Import this in all tasks
config_loader = ConfigLoader()

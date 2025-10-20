#!/usr/bin/env python3
"""
Configuration Management System for Paul Wurth Industrial Time Series Analysis

This module provides centralized configuration management with the following features:
- YAML-based configuration files with hierarchical override
- Environment variable substitution
- Runtime configuration validation
- Type-safe configuration access
- Default value handling

Configuration precedence (highest to lowest):
1. Environment variables (PAUL_WURTH_*)
2. local.yaml configuration file
3. default.yaml configuration file
4. Built-in defaults
"""

import os
import sys
import yaml
import logging
from pathlib import Path
from typing import Any, Dict, Optional, Union, List
from dataclasses import dataclass, field
import warnings

# Configure logging
logger = logging.getLogger(__name__)

class ConfigurationError(Exception):
    """Raised when configuration is invalid or missing"""
    pass

class ConfigManager:
    """Centralized configuration manager for Paul Wurth platform"""

    def __init__(self, config_dir: Optional[Path] = None, environment_prefix: str = "PAUL_WURTH"):
        """
        Initialize configuration manager

        Args:
            config_dir: Directory containing configuration files (default: ./config)
            environment_prefix: Prefix for environment variables (default: PAUL_WURTH)
        """
        self.environment_prefix = environment_prefix
        self.config_dir = config_dir or Path(__file__).parent
        self._config_cache = {}
        self._load_config()

    def _load_config(self):
        """Load configuration from files and environment variables"""
        # Start with default configuration
        default_config_path = self.config_dir / "default.yaml"
        if default_config_path.exists():
            with open(default_config_path, 'r') as f:
                self._config_cache = yaml.safe_load(f) or {}
        else:
            logger.warning(f"Default configuration file not found: {default_config_path}")
            self._config_cache = {}

        # Override with local configuration
        local_config_path = self.config_dir / "local.yaml"
        if local_config_path.exists():
            with open(local_config_path, 'r') as f:
                local_config = yaml.safe_load(f) or {}
                self._config_cache = self._merge_configs(self._config_cache, local_config)
            logger.info(f"Loaded local configuration from {local_config_path}")

        # Override with environment variables
        self._apply_environment_overrides()

        # Validate configuration
        self._validate_config()

        logger.info("Configuration loaded successfully")

    def _merge_configs(self, base: Dict, override: Dict) -> Dict:
        """Recursively merge configuration dictionaries"""
        result = base.copy()

        for key, value in override.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._merge_configs(result[key], value)
            else:
                result[key] = value

        return result

    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        env_prefix = f"{self.environment_prefix}_"

        for env_var, value in os.environ.items():
            if env_var.startswith(env_prefix):
                # Convert PAUL_WURTH_SYSTEM_GPU_ENABLED -> system.gpu_enabled
                config_path = env_var[len(env_prefix):].lower().replace('_', '.')
                self._set_nested_config(config_path, self._parse_env_value(value))

    def _set_nested_config(self, path: str, value: Any):
        """Set nested configuration value using dot notation"""
        keys = path.split('.')
        config = self._config_cache

        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]

        config[keys[-1]] = value

    def _parse_env_value(self, value: str) -> Any:
        """Parse environment variable value to appropriate type"""
        # Handle boolean values
        if value.lower() in ('true', 'yes', '1', 'on'):
            return True
        elif value.lower() in ('false', 'no', '0', 'off'):
            return False

        # Handle numeric values
        try:
            if '.' in value:
                return float(value)
            else:
                return int(value)
        except ValueError:
            pass

        # Handle lists (comma-separated)
        if ',' in value:
            return [item.strip() for item in value.split(',')]

        # Default to string
        return value

    def _validate_config(self):
        """Validate configuration values"""
        # Required sections
        required_sections = ['system', 'data', 'results', 'preprocessing', 'dynotears']
        missing_sections = [section for section in required_sections
                          if section not in self._config_cache]

        if missing_sections:
            logger.warning(f"Missing configuration sections: {missing_sections}")

        # Validate paths
        self._validate_paths()

        # Validate numeric ranges
        self._validate_numeric_ranges()

    def _validate_paths(self):
        """Validate and create necessary paths"""
        path_configs = [
            ('data.base_path', True),
            ('results.base_path', True),
            ('system.temp_dir', True)
        ]

        for path_config, create_if_missing in path_configs:
            path_value = self.get(path_config)
            if path_value:
                path_obj = Path(path_value)
                if create_if_missing and not path_obj.exists():
                    try:
                        path_obj.mkdir(parents=True, exist_ok=True)
                        logger.info(f"Created directory: {path_obj}")
                    except OSError as e:
                        logger.error(f"Failed to create directory {path_obj}: {e}")

    def _validate_numeric_ranges(self):
        """Validate numeric configuration values are in valid ranges"""
        validations = [
            ('dynotears.lambda_w', 0.0, 1.0),
            ('dynotears.lambda_a', 0.0, 1.0),
            ('dynotears.max_iter', 1, 10000),
            ('dynotears.learning_rate', 1e-6, 1.0),
            ('preprocessing.stationarity.alpha', 0.01, 0.5),
            ('system.num_workers', 1, 128),
        ]

        for config_path, min_val, max_val in validations:
            value = self.get(config_path)
            if value is not None:
                if not (min_val <= value <= max_val):
                    logger.warning(f"Configuration {config_path}={value} outside recommended range [{min_val}, {max_val}]")

    def get(self, path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation

        Args:
            path: Configuration path (e.g., 'system.gpu_enabled')
            default: Default value if path not found

        Returns:
            Configuration value or default
        """
        keys = path.split('.')
        value = self._config_cache

        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default

    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.get(section, {})

    def set(self, path: str, value: Any):
        """Set configuration value at runtime"""
        self._set_nested_config(path, value)

    def reload(self):
        """Reload configuration from files"""
        self._config_cache.clear()
        self._load_config()

    def export_config(self, path: Optional[Path] = None) -> Dict[str, Any]:
        """Export current configuration to file or return as dict"""
        if path:
            with open(path, 'w') as f:
                yaml.dump(self._config_cache, f, default_flow_style=False, indent=2)
            logger.info(f"Configuration exported to {path}")

        return self._config_cache.copy()

    def get_effective_config(self) -> Dict[str, Any]:
        """Get the complete effective configuration (for debugging)"""
        return self._config_cache.copy()


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None

def get_config_manager() -> ConfigManager:
    """Get the global configuration manager instance"""
    global _config_manager
    if _config_manager is None:
        _config_manager = ConfigManager()
    return _config_manager

def get_config(path: str, default: Any = None) -> Any:
    """Convenience function to get configuration value"""
    return get_config_manager().get(path, default)

def get_config_section(section: str) -> Dict[str, Any]:
    """Convenience function to get configuration section"""
    return get_config_manager().get_section(section)


# Configuration dataclasses for type-safe access
@dataclass
class SystemConfig:
    """System configuration"""
    gpu_enabled: bool = True
    num_workers: int = 4
    memory_limit_gb: int = 32
    temp_dir: str = "/tmp/paulwurth"
    log_level: str = "INFO"
    log_file: str = "logs/paulwurth.log"

    @classmethod
    def from_config(cls) -> 'SystemConfig':
        config = get_config_section('system')
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})

@dataclass
class DynoTearsConfig:
    """DynoTears algorithm configuration"""
    lambda_w: float = 0.1
    lambda_a: float = 0.1
    max_iter: int = 100
    h_tol: float = 1e-8
    loss_tol: float = 1e-6
    hidden_layers: List[int] = field(default_factory=lambda: [10])
    activation: str = "relu"
    optimizer: str = "adam"
    learning_rate: float = 0.001
    w_threshold: float = 0.0
    device: str = "auto"
    dtype: str = "float32"

    @classmethod
    def from_config(cls) -> 'DynoTearsConfig':
        config = get_config_section('dynotears')
        return cls(**{k: v for k, v in config.items() if k in cls.__annotations__})

@dataclass
class PreprocessingConfig:
    """Preprocessing configuration"""
    alpha: float = field(default=0.05, metadata={'config_path': 'stationarity.alpha'})
    max_lags: int = field(default=10, metadata={'config_path': 'lags.max_lags'})
    mi_alpha: float = field(default=0.01, metadata={'config_path': 'mutual_information.alpha'})
    mi_bins: int = field(default=5, metadata={'config_path': 'mutual_information.bins'})
    parallel_enabled: bool = field(default=True, metadata={'config_path': 'parallel.enabled'})

    @classmethod
    def from_config(cls) -> 'PreprocessingConfig':
        config = get_config_section('preprocessing')
        result = {}

        for field_name, field_info in cls.__annotations__.items():
            if hasattr(cls, field_name):
                field_obj = getattr(cls, field_name)
                if hasattr(field_obj, 'metadata') and 'config_path' in field_obj.metadata:
                    config_path = field_obj.metadata['config_path']
                    result[field_name] = get_config(f'preprocessing.{config_path}', field_obj.default)
                else:
                    result[field_name] = config.get(field_name, field_obj.default if hasattr(field_obj, 'default') else None)

        return cls(**result)


# Command-line configuration utility
def main():
    """Command-line utility for configuration management"""
    import argparse

    parser = argparse.ArgumentParser(description="Paul Wurth Configuration Manager")
    parser.add_argument('--show', action='store_true', help='Show current configuration')
    parser.add_argument('--validate', action='store_true', help='Validate configuration')
    parser.add_argument('--export', type=str, help='Export configuration to file')
    parser.add_argument('--get', type=str, help='Get specific configuration value')
    parser.add_argument('--set', type=str, nargs=2, metavar=('PATH', 'VALUE'),
                       help='Set configuration value (path value)')

    args = parser.parse_args()

    config_manager = get_config_manager()

    if args.show:
        print("Current Configuration:")
        print(yaml.dump(config_manager.get_effective_config(), default_flow_style=False))

    if args.validate:
        try:
            config_manager._validate_config()
            print("Configuration is valid ✓")
        except Exception as e:
            print(f"Configuration validation failed: {e}")
            sys.exit(1)

    if args.export:
        config_manager.export_config(Path(args.export))
        print(f"Configuration exported to {args.export}")

    if args.get:
        value = config_manager.get(args.get)
        print(f"{args.get}: {value}")

    if args.set:
        path, value = args.set
        parsed_value = config_manager._parse_env_value(value)
        config_manager.set(path, parsed_value)
        print(f"Set {path} = {parsed_value}")

if __name__ == "__main__":
    main()
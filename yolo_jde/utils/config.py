"""Configuration management utilities."""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional


class ConfigManager:
    """Configuration manager for YOLO-JDE tracker."""
    
    def __init__(self, config_dir: Optional[str] = None):
        """Initialize configuration manager.
        
        Args:
            config_dir: Path to configuration directory. If None, uses package configs.
        """
        if config_dir is None:
            # Use package configs
            self.config_dir = Path(__file__).parent.parent.parent / "configs"
        else:
            self.config_dir = Path(config_dir)
    
    def get_tracker_config_path(self, tracker_name: str) -> Path:
        """Get path to tracker configuration file.
        
        Args:
            tracker_name: Name of tracker (e.g., 'smiletrack', 'bytetrack')
            
        Returns:
            Path object to tracker configuration file
        """
        return self.config_dir / "trackers" / f"{tracker_name}.yaml"
    
    def load_tracker_config(self, tracker_name: str) -> Dict[str, Any]:
        """Load tracker configuration from YAML file.
        
        Args:
            tracker_name: Name of tracker (e.g., 'smiletrack', 'bytetrack')
            
        Returns:
            Dictionary containing tracker configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.get_tracker_config_path(tracker_name)
        if not config_path.exists():
            raise FileNotFoundError(f"Tracker config not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def load_model_config(self, model_name: str) -> Dict[str, Any]:
        """Load model configuration from YAML file.
        
        Args:
            model_name: Name of model configuration
            
        Returns:
            Dictionary containing model configuration
            
        Raises:
            FileNotFoundError: If config file doesn't exist
        """
        config_path = self.config_dir / "models" / f"{model_name}.yaml"
        if not config_path.exists():
            raise FileNotFoundError(f"Model config not found: {config_path}")
            
        with open(config_path, 'r', encoding='utf-8') as f:
            return yaml.safe_load(f)
    
    def save_config(self, config: Dict[str, Any], config_type: str, name: str) -> None:
        """Save configuration to YAML file.
        
        Args:
            config: Configuration dictionary to save
            config_type: Type of config ('trackers' or 'models')
            name: Name of configuration file (without extension)
        """
        config_dir = self.config_dir / config_type
        config_dir.mkdir(parents=True, exist_ok=True)
        
        config_path = config_dir / f"{name}.yaml"
        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    
    def list_configs(self, config_type: str) -> list:
        """List available configurations.
        
        Args:
            config_type: Type of config ('trackers' or 'models')
            
        Returns:
            List of available configuration names
        """
        config_dir = self.config_dir / config_type
        if not config_dir.exists():
            return []
            
        return [f.stem for f in config_dir.glob("*.yaml")]

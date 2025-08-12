"""
Dynamic Configuration Manager for Phase 3 optimization.
Provides runtime configuration updates, hot-reload capabilities, and configuration validation.
"""

import asyncio
import json
import os
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from pathlib import Path
import logging

from .config_validator import ConfigValidator
from .language_config.base_config import LanguageConfigFactory


class DynamicConfigManager:
    """Dynamic configuration manager with runtime updates and validation."""
    
    def __init__(self, config_dir: str = "src/config"):
        self.config_dir = Path(config_dir)
        self.config_watchers: Dict[str, List[Callable]] = {}
        self.config_backups: Dict[str, Dict] = {}
        self.config_cache: Dict[str, Any] = {}
        self.validator = ConfigValidator()
        self.logger = logging.getLogger(__name__)
        
        # Initialize configuration monitoring
        self._setup_config_monitoring()
    
    def _setup_config_monitoring(self):
        """Setup configuration file monitoring."""
        try:
            # Create config directory if it doesn't exist
            self.config_dir.mkdir(parents=True, exist_ok=True)
            
            # Initialize backup directory
            backup_dir = self.config_dir / "backups"
            backup_dir.mkdir(exist_ok=True)
            
            self.logger.info(f"DynamicConfigManager initialized with config directory: {self.config_dir}")
            
        except Exception as e:
            self.logger.error(f"Error setting up config monitoring: {e}")
    
    async def update_language_config(self, language_code: str, new_config: dict) -> bool:
        """Update language configuration at runtime."""
        try:
            # Validate new configuration
            if not self.validator.validate_language_config(new_config):
                raise ValueError(f"Invalid language configuration for {language_code}")
            
            # Backup current configuration
            await self._backup_config(language_code)
            
            # Update configuration
            success = await self._apply_config_update(language_code, new_config)
            
            if success:
                # Notify watchers
                await self._notify_config_watchers(language_code, new_config)
                
                # Update cache
                self.config_cache[language_code] = new_config
                
                self.logger.info(f"Successfully updated configuration for {language_code}")
                return True
            else:
                # Restore from backup if update failed
                await self._restore_config(language_code)
                return False
                
        except Exception as e:
            self.logger.error(f"Error updating language config for {language_code}: {e}")
            return False
    
    async def hot_reload_config(self, config_file: str) -> bool:
        """Hot reload configuration from file."""
        try:
            config_path = self.config_dir / config_file
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load new configuration
            new_config = await self._load_config_from_file(config_path)
            
            # Update each language configuration
            success_count = 0
            total_count = len(new_config)
            
            for language_code, config in new_config.items():
                if await self.update_language_config(language_code, config):
                    success_count += 1
            
            self.logger.info(f"Hot reload completed: {success_count}/{total_count} configurations updated")
            return success_count == total_count
            
        except Exception as e:
            self.logger.error(f"Error during hot reload: {e}")
            return False
    
    async def get_config_status(self) -> Dict[str, Any]:
        """Get status of all configurations."""
        try:
            status = {
                "total_languages": len(self.config_cache),
                "backup_count": len(self.config_backups),
                "watcher_count": sum(len(watchers) for watchers in self.config_watchers.values()),
                "last_update": None,
                "configs": {}
            }
            
            for lang_code, config in self.config_cache.items():
                status["configs"][lang_code] = {
                    "has_backup": lang_code in self.config_backups,
                    "has_watchers": lang_code in self.config_watchers,
                    "config_size": len(str(config)),
                    "last_modified": self._get_config_last_modified(lang_code)
                }
            
            return status
            
        except Exception as e:
            self.logger.error(f"Error getting config status: {e}")
            return {"error": str(e)}
    
    def add_config_watcher(self, language_code: str, callback: Callable):
        """Add a watcher for configuration changes."""
        if language_code not in self.config_watchers:
            self.config_watchers[language_code] = []
        
        self.config_watchers[language_code].append(callback)
        self.logger.info(f"Added config watcher for {language_code}")
    
    def remove_config_watcher(self, language_code: str, callback: Callable):
        """Remove a configuration watcher."""
        if language_code in self.config_watchers:
            try:
                self.config_watchers[language_code].remove(callback)
                self.logger.info(f"Removed config watcher for {language_code}")
            except ValueError:
                self.logger.warning(f"Watcher not found for {language_code}")
    
    async def _backup_config(self, language_code: str):
        """Backup current configuration."""
        try:
            current_config = self._get_current_config(language_code)
            if current_config:
                backup_data = {
                    "language_code": language_code,
                    "config": current_config,
                    "backup_time": datetime.now().isoformat(),
                    "version": "1.0"
                }
                
                self.config_backups[language_code] = backup_data
                
                # Save to file
                backup_file = self.config_dir / "backups" / f"{language_code}_backup_{int(time.time())}.json"
                with open(backup_file, 'w', encoding='utf-8') as f:
                    json.dump(backup_data, f, ensure_ascii=False, indent=2)
                
                self.logger.info(f"Backed up configuration for {language_code}")
                
        except Exception as e:
            self.logger.error(f"Error backing up config for {language_code}: {e}")
    
    async def _restore_config(self, language_code: str) -> bool:
        """Restore configuration from backup."""
        try:
            if language_code in self.config_backups:
                backup_data = self.config_backups[language_code]
                await self._apply_config_update(language_code, backup_data["config"])
                self.logger.info(f"Restored configuration for {language_code}")
                return True
            else:
                self.logger.warning(f"No backup found for {language_code}")
                return False
                
        except Exception as e:
            self.logger.error(f"Error restoring config for {language_code}: {e}")
            return False
    
    def _get_current_config(self, language_code: str) -> Optional[Dict]:
        """Get current configuration for a language."""
        try:
            # Try to get from cache first
            if language_code in self.config_cache:
                return self.config_cache[language_code]
            
            # Try to get from language config factory
            config = LanguageConfigFactory.get_config(language_code)
            if config:
                return {
                    "entity_patterns": config.entity_patterns.__dict__ if hasattr(config, 'entity_patterns') else {},
                    "processing_settings": config.processing_settings.__dict__ if hasattr(config, 'processing_settings') else {},
                    "detection_patterns": config.detection_patterns if hasattr(config, 'detection_patterns') else {},
                    "language_code": language_code
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"Error getting current config for {language_code}: {e}")
            return None
    
    async def _apply_config_update(self, language_code: str, new_config: dict) -> bool:
        """Apply configuration update."""
        try:
            # Update the language configuration
            if hasattr(LanguageConfigFactory, 'update_config'):
                await LanguageConfigFactory.update_config(language_code, new_config)
            
            # Update cache
            self.config_cache[language_code] = new_config
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error applying config update for {language_code}: {e}")
            return False
    
    async def _notify_config_watchers(self, language_code: str, new_config: dict):
        """Notify all watchers of configuration change."""
        if language_code in self.config_watchers:
            for callback in self.config_watchers[language_code]:
                try:
                    if asyncio.iscoroutinefunction(callback):
                        await callback(language_code, new_config)
                    else:
                        callback(language_code, new_config)
                except Exception as e:
                    self.logger.error(f"Error in config watcher callback: {e}")
    
    async def _load_config_from_file(self, config_path: Path) -> Dict[str, Any]:
        """Load configuration from file."""
        try:
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
            
            return config_data
            
        except Exception as e:
            self.logger.error(f"Error loading config from file {config_path}: {e}")
            raise
    
    def _get_config_last_modified(self, language_code: str) -> Optional[str]:
        """Get last modified time for configuration."""
        try:
            config_file = self.config_dir / f"{language_code}_config.json"
            if config_file.exists():
                return datetime.fromtimestamp(config_file.stat().st_mtime).isoformat()
            return None
        except Exception:
            return None


# Global instance
dynamic_config_manager = DynamicConfigManager()

"""
Channel naming and management system for SPT Analysis.
Provides persistent storage and retrieval of biological channel names.
"""

import json
import os
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class ChannelManager:
    """Manages channel naming with persistent global storage."""
    
    MAX_NAME_LENGTH = 24
    
    def __init__(self, config_dir: str = None):
        if config_dir is None:
            config_dir = os.path.expanduser("~/.spt2025b")
        
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(exist_ok=True)
        self.channel_names_file = self.config_dir / "channel_names.json"
        self.load_channel_names()
    
    def load_channel_names(self):
        """Load saved channel names from disk."""
        if self.channel_names_file.exists():
            try:
                with open(self.channel_names_file, 'r') as f:
                    self.saved_names = json.load(f)
            except (json.JSONDecodeError, FileNotFoundError):
                self.saved_names = []
        else:
            self.saved_names = []
    
    def save_channel_names(self):
        """Save channel names to disk."""
        try:
            with open(self.channel_names_file, 'w') as f:
                json.dump(self.saved_names, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save channel names: {e}")
    
    def validate_name(self, name: str) -> Tuple[bool, str]:
        """Validate channel name against constraints."""
        if not name or not name.strip():
            return False, "Channel name cannot be empty"
        
        name = name.strip()
        if len(name) > self.MAX_NAME_LENGTH:
            return False, f"Channel name must be {self.MAX_NAME_LENGTH} characters or less"
        
        return True, name
    
    def add_channel_name(self, name: str) -> bool:
        """Add a new channel name to the saved list."""
        is_valid, validated_name = self.validate_name(name)
        if not is_valid:
            return False
        
        if validated_name not in self.saved_names:
            self.saved_names.append(validated_name)
            self.save_channel_names()
        return True
    
    def get_common_names(self) -> List[str]:
        """Get list of commonly used channel names."""
        return sorted(self.saved_names)
    
    def get_suggestions(self, partial_name: str = "") -> List[str]:
        """Get channel name suggestions based on partial input."""
        if not partial_name:
            return self.get_common_names()
        
        partial_lower = partial_name.lower()
        suggestions = [name for name in self.saved_names 
                      if partial_lower in name.lower()]
        return sorted(suggestions)

channel_manager = ChannelManager()

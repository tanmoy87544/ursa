"""
Memory components for the scientific agent framework.
Handles storing, retrieving, and managing agent memory.
"""

import logging
import json
import os
import time
from typing import Dict, Any, List, Optional, Union

logger = logging.getLogger(__name__)

class Memory:
    """Memory system for scientific problem-solving."""
    
    def __init__(self, memory_file: Optional[str] = None):
        """
        Initialize the memory system.
        
        Args:
            memory_file: Optional file path for persisting memory
        """
        self.memory_file = memory_file
        
        # Initialize memory containers
        self.short_term = {}  # Current problem context
        self.long_term = {}   # Persistent knowledge
        self.episodic = []    # History of actions and results
        
        # Load existing memory if available
        if memory_file and os.path.exists(memory_file):
            self._load_memory()
    
    def add(self, key: str, value: Any) -> None:
        """
        Add information to short-term memory.
        
        Args:
            key: Key for storing the information
            value: Information to store
        """
        self.short_term[key] = value
        
        # Add to episodic memory with timestamp
        self.episodic.append({
            "action": "add",
            "key": key,
            "timestamp": time.time()
        })
        
        # Persist if memory file is set
        if self.memory_file:
            self._save_memory()
    
    def retrieve(self, key: str) -> Any:
        """
        Retrieve information from memory.
        
        Args:
            key: Key to retrieve
            
        Returns:
            Retrieved information or None if not found
        """
        # Add to episodic memory
        self.episodic.append({
            "action": "retrieve",
            "key": key,
            "timestamp": time.time()
        })
        
        # Check short-term memory first
        if key in self.short_term:
            return self.short_term[key]
        
        # Then check long-term memory
        if key in self.long_term:
            return self.long_term[key]
        
        return None
    
    def persist(self, key: str) -> bool:
        """
        Move information from short-term to long-term memory.
        
        Args:
            key: Key to persist
            
        Returns:
            True if successfully persisted, False otherwise
        """
        if key in self.short_term:
            # Move to long-term memory
            self.long_term[key] = self.short_term[key]
            
            # Add to episodic memory
            self.episodic.append({
                "action": "persist",
                "key": key,
                "timestamp": time.time()
            })
            
            # Persist if memory file is set
            if self.memory_file:
                self._save_memory()
            
            return True
        
        return False
    
    def forget(self, key: str) -> bool:
        """
        Remove information from short-term memory.
        
        Args:
            key: Key to forget
            
        Returns:
            True if successfully forgotten, False otherwise
        """
        if key in self.short_term:
            # Remove from short-term memory
            del self.short_term[key]
            
            # Add to episodic memory
            self.episodic.append({
                "action": "forget",
                "key": key,
                "timestamp": time.time()
            })
            
            # Persist if memory file is set
            if self.memory_file:
                self._save_memory()
            
            return True
        
        return False
    
    def clear_short_term(self) -> None:
        """Clear all short-term memory."""
        self.short_term = {}
        
        # Add to episodic memory
        self.episodic.append({
            "action": "clear_short_term",
            "timestamp": time.time()
        })
        
        # Persist if memory file is set
        if self.memory_file:
            self._save_memory()
    
    def get_all_short_term(self) -> Dict[str, Any]:
        """Get all short-term memory items."""
        return self.short_term.copy()
    
    def get_all_long_term(self) -> Dict[str, Any]:
        """Get all long-term memory items."""
        return self.long_term.copy()
    
    def get_episodic(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get episodic memory entries.
        
        Args:
            limit: Optional limit on number of entries to return
            
        Returns:
            List of episodic memory entries
        """
        if limit:
            return self.episodic[-limit:]
        return self.episodic.copy()
    
    def search(self, query: str) -> Dict[str, List[Any]]:
        """
        Search memory for items matching the query.
        
        Args:
            query: Search query string
            
        Returns:
            Dictionary of matching items from short-term and long-term memory
        """
        # Simple string matching search - in a real system, this would be more sophisticated
        short_term_matches = []
        long_term_matches = []
        
        query = query.lower()
        
        # Search short-term memory
        for key, value in self.short_term.items():
            if query in key.lower():
                short_term_matches.append((key, value))
            elif isinstance(value, str) and query in value.lower():
                short_term_matches.append((key, value))
            elif isinstance(value, dict):
                # Search for query in dict values that are strings
                for k, v in value.items():
                    if isinstance(v, str) and query in v.lower():
                        short_term_matches.append((key, value))
                        break
        
        # Search long-term memory
        for key, value in self.long_term.items():
            if query in key.lower():
                long_term_matches.append((key, value))
            elif isinstance(value, str) and query in value.lower():
                long_term_matches.append((key, value))
            elif isinstance(value, dict):
                # Search for query in dict values that are strings
                for k, v in value.items():
                    if isinstance(v, str) and query in v.lower():
                        long_term_matches.append((key, value))
                        break
        
        # Add to episodic memory
        self.episodic.append({
            "action": "search",
            "query": query,
            "short_term_matches": len(short_term_matches),
            "long_term_matches": len(long_term_matches),
            "timestamp": time.time()
        })
        
        return {
            "short_term": short_term_matches,
            "long_term": long_term_matches
        }
    
    def _json_serializable(self, obj: Any) -> Any:
        """Convert an object to a JSON serializable format."""
        if hasattr(obj, 'to_dict'):
            return obj.to_dict()
        elif hasattr(obj, '__dict__'):
            return obj.__dict__
        else:
            return str(obj)
    
    def _save_memory(self) -> None:
        """Save memory to file."""
        try:
            memory_data = {
                "short_term": self.short_term,
                "long_term": self.long_term,
                "episodic": self.episodic,
                "last_saved": time.time()
            }
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(os.path.abspath(self.memory_file)), exist_ok=True)
            
            with open(self.memory_file, 'w') as f:
                json.dump(memory_data, f, default=self._json_serializable, indent=2)
                
            logger.debug(f"Memory saved to {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Error saving memory: {str(e)}")
    
    def _load_memory(self) -> None:
        """Load memory from file."""
        try:
            with open(self.memory_file, 'r') as f:
                memory_data = json.load(f)
            
            self.short_term = memory_data.get("short_term", {})
            self.long_term = memory_data.get("long_term", {})
            self.episodic = memory_data.get("episodic", [])
            
            logger.debug(f"Memory loaded from {self.memory_file}")
            
        except Exception as e:
            logger.error(f"Error loading memory: {str(e)}")
            # Initialize with empty memory
            self.short_term = {}
            self.long_term = {}
            self.episodic = []

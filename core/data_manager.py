"""
DataManager singleton class for managing multiple datasets in DOODA.

Handles multiple loaded files, derived datasets from processing,
and virtual bands with full lineage tracking.
"""

from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from datetime import datetime
import weakref
from pathlib import Path

from data_handler import DataHandler


class DataManager:
    """Singleton manager for all datasets in the application."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
            
        self._initialized = True
        self.datasets: Dict[str, DataHandler] = {}
        self.metadata: Dict[str, dict] = {}
        self.active_dataset: Optional[str] = None
        
        # Track derived datasets with weak references to save memory
        self.derived_refs: Dict[str, weakref.ref] = {}
        
    def add_dataset(self, name: str, data_handler: DataHandler, 
                    source_info: Optional[dict] = None) -> bool:
        """
        Add a new dataset to the manager.
        
        Args:
            name: Unique name for the dataset
            data_handler: DataHandler instance with loaded data
            source_info: Optional metadata about data source and processing
            
        Returns:
            True if added successfully, False if name already exists
        """
        if name in self.datasets:
            # Append timestamp to make unique
            timestamp = datetime.now().strftime("%H%M%S")
            name = f"{name}_{timestamp}"
            
        self.datasets[name] = data_handler
        
        # Store metadata
        self.metadata[name] = {
            'added_time': datetime.now().isoformat(),
            'source_info': source_info or {},
            'type': 'original' if not source_info else source_info.get('type', 'derived'),
            'parent_datasets': source_info.get('parents', []) if source_info else [],
            'processing_history': source_info.get('processing', []) if source_info else []
        }
        
        # Set as active if it's the first dataset
        if self.active_dataset is None:
            self.active_dataset = name
            
        return True
        
    def get_dataset(self, name: str) -> Optional[DataHandler]:
        """Get a dataset by name."""
        return self.datasets.get(name)
        
    def get_active_dataset(self) -> Optional[DataHandler]:
        """Get the currently active dataset."""
        if self.active_dataset:
            return self.datasets.get(self.active_dataset)
        return None
        
    def set_active_dataset(self, name: str) -> bool:
        """Set the active dataset."""
        if name in self.datasets:
            self.active_dataset = name
            return True
        return False
        
    def create_derived(self, name: str, data: np.ndarray, 
                      parent_datasets: List[str],
                      operation: str,
                      wavelengths: Optional[np.ndarray] = None) -> Optional[str]:
        """
        Create a derived dataset from processing operations.
        
        Args:
            name: Name for the new dataset
            data: Processed data array
            parent_datasets: List of parent dataset names used
            operation: Description of the operation performed
            wavelengths: Optional wavelength array for the new data
            
        Returns:
            Name of the created dataset or None if failed
        """
        # Create new DataHandler for derived data
        derived_handler = DataHandler()
        
        # Set data directly (bypassing file loading)
        derived_handler.data = data
        derived_handler.shape = data.shape
        derived_handler.wavelengths = wavelengths
        derived_handler.is_loaded = True
        derived_handler.filename = f"derived:{name}"
        derived_handler.file_type = 'derived'
        
        # Copy header info from first parent if available
        if parent_datasets and parent_datasets[0] in self.datasets:
            parent = self.datasets[parent_datasets[0]]
            if hasattr(parent, 'header'):
                derived_handler.header = parent.header.copy()
                derived_handler.header['description'] = operation
        
        # Add to manager with lineage info
        source_info = {
            'type': 'derived',
            'parents': parent_datasets,
            'processing': [{
                'operation': operation,
                'timestamp': datetime.now().isoformat()
            }]
        }
        
        self.add_dataset(name, derived_handler, source_info)
        return name
        
    def remove_dataset(self, name: str) -> bool:
        """Remove a dataset from the manager."""
        if name in self.datasets:
            del self.datasets[name]
            del self.metadata[name]
            
            # Update active dataset if needed
            if self.active_dataset == name:
                self.active_dataset = list(self.datasets.keys())[0] if self.datasets else None
                
            return True
        return False
        
    def list_datasets(self) -> List[dict]:
        """
        List all datasets with their metadata.
        
        Returns:
            List of dataset info dictionaries
        """
        result = []
        for name, handler in self.datasets.items():
            info = {
                'name': name,
                'shape': handler.shape if handler.is_loaded else None,
                'type': self.metadata[name]['type'],
                'parents': self.metadata[name]['parent_datasets'],
                'added_time': self.metadata[name]['added_time'],
                'is_active': name == self.active_dataset
            }
            result.append(info)
        return result
        
    def clear_all(self):
        """Clear all datasets from memory."""
        self.datasets.clear()
        self.metadata.clear()
        self.derived_refs.clear()
        self.active_dataset = None
        
    def get_dataset_info(self, name: str) -> Optional[dict]:
        """Get detailed information about a dataset."""
        if name not in self.datasets:
            return None
            
        handler = self.datasets[name]
        meta = self.metadata[name]
        
        return {
            'name': name,
            'shape': handler.shape,
            'dtype': str(handler.data_type) if handler.data_type else 'unknown',
            'memory_size': handler.data.nbytes if handler.data is not None else 0,
            'wavelengths': handler.wavelengths is not None,
            'num_wavelengths': len(handler.wavelengths) if handler.wavelengths is not None else 0,
            'type': meta['type'],
            'parents': meta['parent_datasets'],
            'processing_history': meta['processing_history'],
            'added_time': meta['added_time'],
            'file_path': handler.filename if handler.filename != f"derived:{name}" else None
        }
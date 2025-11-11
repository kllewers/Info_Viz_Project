"""
ROIManager class for ROI handling and statistics computation.

Manages ROI definitions, computes vectorized statistics, and handles export/import.
"""

import numpy as np
import json
import yaml
import os
import datetime
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass


@dataclass
class ROIStats:
    """Container for ROI statistics."""
    mean: np.ndarray
    std: np.ndarray
    min: np.ndarray
    max: np.ndarray
    count: int
    mask: np.ndarray


class ROIManager:
    """Manages ROI definitions and computes statistics efficiently."""
    
    def __init__(self):
        self.rois = {}  # roi_id -> ROI definition
        self.roi_stats = {}  # roi_id -> ROIStats
        self.data_handler = None
        
    def set_data_handler(self, data_handler):
        """Set reference to DataHandler for spectrum computation."""
        self.data_handler = data_handler
        
    def add_roi(self, roi_id: str, roi_definition: Dict[str, Any], 
                name: Optional[str] = None) -> bool:
        """
        Add a new ROI.
        
        Args:
            roi_id: Unique identifier for ROI
            roi_definition: ROI definition (currently supports rectangular)
            name: Optional human-readable name
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Validate ROI definition
            if not self._validate_roi_definition(roi_definition):
                return False
                
            roi_info = {
                'definition': roi_definition,
                'name': name or f'ROI {roi_id}',
                'created': self._get_timestamp()
            }
            
            self.rois[roi_id] = roi_info
            
            # Compute statistics if data is available
            if self.data_handler is not None and self.data_handler.is_loaded:
                self._compute_roi_stats(roi_id)
                
            return True
            
        except Exception as e:
            print(f"Error adding ROI {roi_id}: {e}")
            return False
            
    def remove_roi(self, roi_id: str) -> bool:
        """Remove ROI and its statistics."""
        try:
            if roi_id in self.rois:
                del self.rois[roi_id]
                
            if roi_id in self.roi_stats:
                del self.roi_stats[roi_id]
                
            return True
            
        except Exception as e:
            print(f"Error removing ROI {roi_id}: {e}")
            return False
            
    def clear_all_rois(self):
        """Clear all ROIs and statistics."""
        self.rois.clear()
        self.roi_stats.clear()
        
    def get_roi_list(self) -> List[str]:
        """Get list of ROI IDs."""
        return list(self.rois.keys())
        
    def get_roi_info(self, roi_id: str) -> Optional[Dict[str, Any]]:
        """Get ROI information."""
        return self.rois.get(roi_id)
        
    def get_roi_stats(self, roi_id: str) -> Optional[ROIStats]:
        """Get ROI statistics."""
        return self.roi_stats.get(roi_id)
        
    def _validate_roi_definition(self, roi_def: Dict[str, Any]) -> bool:
        """Validate ROI definition format."""
        # Support both legacy format (x, y, width, height) and new format (points, type)
        legacy_keys = ['x', 'y', 'width', 'height']
        new_format_keys = ['points', 'type']
        
        has_legacy = all(key in roi_def for key in legacy_keys)
        has_new = all(key in roi_def for key in new_format_keys)
        
        return has_legacy or has_new
        
    def _get_timestamp(self) -> str:
        """Get current timestamp string."""
        return datetime.datetime.now().isoformat()
        
    def _compute_roi_stats(self, roi_id: str):
        """Compute statistics for ROI."""
        if roi_id not in self.rois or self.data_handler is None:
            return
            
        try:
            roi_def = self.rois[roi_id]['definition']
            
            # Check if we have direct points or need to create mask
            if 'points' in roi_def and roi_def.get('type') in ['point', 'rectangle', 'polygon']:
                # New format with direct point coordinates
                points = roi_def['points']
                spectra = self._extract_roi_spectra_from_points(points)
                
                # Create mask from points for compatibility
                mask = self._create_mask_from_points(points) if points else None
                
            else:
                # Legacy format - create mask from rectangle definition
                mask = self._create_roi_mask(roi_def)
                if mask is None:
                    return
                spectra = self._extract_roi_spectra(mask)
                
            if spectra is None or spectra.size == 0:
                return
                
            # Compute statistics
            stats = ROIStats(
                mean=np.mean(spectra, axis=0),
                std=np.std(spectra, axis=0),
                min=np.min(spectra, axis=0),
                max=np.max(spectra, axis=0),
                count=spectra.shape[0],
                mask=mask
            )
            
            self.roi_stats[roi_id] = stats
            
        except Exception as e:
            print(f"Error computing ROI stats for {roi_id}: {e}")
            
    def _create_roi_mask(self, roi_def: Dict[str, Any]) -> Optional[np.ndarray]:
        """Create boolean mask for ROI pixels."""
        if self.data_handler is None or not self.data_handler.is_loaded:
            return None
            
        try:
            shape = self.data_handler.shape  # (lines, samples, bands)
            mask = np.zeros((shape[0], shape[1]), dtype=bool)
            
            # Handle rectangular ROI
            x = int(roi_def['x'])
            y = int(roi_def['y'])
            width = int(roi_def['width'])
            height = int(roi_def['height'])
            
            # Ensure bounds are valid
            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(shape[1], x + width)
            y2 = min(shape[0], y + height)
            
            if x2 > x1 and y2 > y1:
                mask[y1:y2, x1:x2] = True
                
            return mask
            
        except Exception as e:
            print(f"Error creating ROI mask: {e}")
            return None
            
    def _extract_roi_spectra(self, mask: np.ndarray) -> Optional[np.ndarray]:
        """Extract spectra for all pixels in ROI mask."""
        if self.data_handler is None or not self.data_handler.is_loaded:
            return None
            
        try:
            # Get pixel coordinates where mask is True
            y_coords, x_coords = np.where(mask)
            
            if len(y_coords) == 0:
                return None
                
            # Extract spectra based on interleave format
            num_bands = self.data_handler.shape[2]
            spectra = np.zeros((len(y_coords), num_bands))
            
            for i, (y, x) in enumerate(zip(y_coords, x_coords)):
                spectrum = self.data_handler.get_pixel_spectrum(x, y)
                if spectrum is not None:
                    spectra[i, :] = spectrum
                    
            return spectra
            
        except Exception as e:
            print(f"Error extracting ROI spectra: {e}")
            return None
            
    def _create_mask_from_points(self, points: list) -> Optional[np.ndarray]:
        """Create boolean mask from list of point coordinates."""
        if self.data_handler is None or not self.data_handler.is_loaded:
            return None
            
        try:
            shape = self.data_handler.shape  # (lines, samples, bands)
            mask = np.zeros((shape[0], shape[1]), dtype=bool)
            
            for x, y in points:
                if 0 <= x < shape[1] and 0 <= y < shape[0]:
                    mask[y, x] = True
                    
            return mask
            
        except Exception as e:
            print(f"Error creating mask from points: {e}")
            return None
            
    def _extract_roi_spectra_from_points(self, points: list) -> Optional[np.ndarray]:
        """Extract spectra for specific pixel coordinates."""
        if self.data_handler is None or not self.data_handler.is_loaded:
            return None
            
        try:
            if len(points) == 0:
                return None
                
            # Extract spectra for each point
            num_bands = self.data_handler.shape[2]
            spectra = np.zeros((len(points), num_bands))
            
            for i, (x, y) in enumerate(points):
                spectrum = self.data_handler.get_pixel_spectrum(x, y)
                if spectrum is not None:
                    spectra[i, :] = spectrum
                    
            return spectra
            
        except Exception as e:
            print(f"Error extracting ROI spectra from points: {e}")
            return None
            
    def update_all_roi_stats(self):
        """Recompute statistics for all ROIs."""
        if self.data_handler is None or not self.data_handler.is_loaded:
            return
            
        for roi_id in self.rois:
            self._compute_roi_stats(roi_id)
            
    def export_rois(self, filename: str, format: str = 'json') -> bool:
        """
        Export ROI definitions to file.
        
        Args:
            filename: Output filename
            format: Export format ('json', 'yaml', or 'envi')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            export_data = {
                'rois': self.rois,
                'metadata': {
                    'export_format': format,
                    'timestamp': self._get_timestamp(),
                    'data_file': self.data_handler.filename if self.data_handler else None,
                    'data_shape': self.data_handler.shape if self.data_handler else None
                }
            }
            
            if format.lower() == 'json':
                with open(filename, 'w') as f:
                    json.dump(export_data, f, indent=2)
                    
            elif format.lower() == 'yaml':
                with open(filename, 'w') as f:
                    yaml.dump(export_data, f, default_flow_style=False)
                    
            elif format.lower() == 'envi':
                self._export_envi_roi(filename)
                
            else:
                print(f"Unsupported export format: {format}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error exporting ROIs: {e}")
            return False
            
    def import_rois(self, filename: str) -> bool:
        """
        Import ROI definitions from file.
        
        Args:
            filename: Input filename
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filename):
                print(f"File not found: {filename}")
                return False
                
            # Detect format from extension
            ext = os.path.splitext(filename)[1].lower()
            
            if ext == '.json':
                with open(filename, 'r') as f:
                    data = json.load(f)
                    
            elif ext in ['.yaml', '.yml']:
                with open(filename, 'r') as f:
                    data = yaml.safe_load(f)
                    
            elif ext == '.roi':
                return self._import_envi_roi(filename)
                
            else:
                print(f"Unsupported import format: {ext}")
                return False
                
            # Import ROI data
            if 'rois' in data:
                for roi_id, roi_info in data['rois'].items():
                    self.rois[roi_id] = roi_info
                    
                # Recompute statistics
                self.update_all_roi_stats()
                
            return True
            
        except Exception as e:
            print(f"Error importing ROIs: {e}")
            return False
            
    def _export_envi_roi(self, filename: str):
        """Export ROIs in ENVI .roi format."""
        # Simplified ENVI ROI export - real implementation would need ENVI format specs
        with open(filename, 'w') as f:
            f.write("; ENVI Region of Interest File\n")
            f.write(f"; Number of ROIs: {len(self.rois)}\n")
            
            for i, (roi_id, roi_info) in enumerate(self.rois.items()):
                roi_def = roi_info['definition']
                f.write(f"; ROI name: {roi_info['name']}\n")
                f.write(f"ROI_1[{i+1}] = {{\n")
                f.write(f"  Color = (255, 0, 0)\n")
                f.write(f"  npts = 4\n")
                
                # Convert rectangle to points
                x, y = roi_def['x'], roi_def['y']
                w, h = roi_def['width'], roi_def['height']
                
                f.write(f"  pts = [{x}, {y}]\n")
                f.write(f"  pts = [{x+w}, {y}]\n")
                f.write(f"  pts = [{x+w}, {y+h}]\n")
                f.write(f"  pts = [{x}, {y+h}]\n")
                f.write("}\n")
                
    def _import_envi_roi(self, filename: str) -> bool:
        """Import ROIs from ENVI .roi format."""
        # Simplified ENVI ROI import - real implementation would need full parser
        try:
            with open(filename, 'r') as f:
                content = f.read()
                
            # Basic parsing for rectangular ROIs
            # This is a simplified version - real ENVI ROI files are more complex
            lines = content.split('\n')
            current_roi = None
            roi_counter = 0
            
            for line in lines:
                line = line.strip()
                
                if line.startswith('ROI_'):
                    roi_counter += 1
                    current_roi = f"imported_{roi_counter}"
                    
                elif line.startswith('pts = [') and current_roi:
                    # Parse point coordinates
                    coords = line.replace('pts = [', '').replace(']', '')
                    x, y = map(int, coords.split(', '))
                    
                    # For first point, initialize ROI
                    if current_roi not in self.rois:
                        self.add_roi(current_roi, {
                            'x': x, 'y': y, 'width': 1, 'height': 1
                        }, f"Imported ROI {roi_counter}")
                        
            return True
            
        except Exception as e:
            print(f"Error importing ENVI ROI: {e}")
            return False
            
    def get_roi_summary(self) -> Dict[str, Any]:
        """Get summary of all ROIs and their statistics."""
        summary = {
            'num_rois': len(self.rois),
            'rois': {}
        }
        
        for roi_id, roi_info in self.rois.items():
            roi_summary = {
                'name': roi_info['name'],
                'definition': roi_info['definition'],
                'has_stats': roi_id in self.roi_stats
            }
            
            if roi_id in self.roi_stats:
                stats = self.roi_stats[roi_id]
                roi_summary['stats'] = {
                    'pixel_count': stats.count,
                    'mean_spectrum_range': (float(np.min(stats.mean)), float(np.max(stats.mean))),
                    'std_spectrum_range': (float(np.min(stats.std)), float(np.max(stats.std)))
                }
                
            summary['rois'][roi_id] = roi_summary
            
        return summary
        
    def create_combined_mask(self, roi_ids: List[str]) -> Optional[np.ndarray]:
        """Create combined mask from multiple ROIs."""
        if not roi_ids or self.data_handler is None:
            return None
            
        try:
            shape = self.data_handler.shape[:2]  # (lines, samples)
            combined_mask = np.zeros(shape, dtype=bool)
            
            for roi_id in roi_ids:
                if roi_id in self.roi_stats:
                    combined_mask |= self.roi_stats[roi_id].mask
                    
            return combined_mask
            
        except Exception as e:
            print(f"Error creating combined mask: {e}")
            return None
            
    def compute_combined_stats(self, roi_ids: List[str]) -> Optional[ROIStats]:
        """Compute statistics for combined ROI."""
        combined_mask = self.create_combined_mask(roi_ids)
        if combined_mask is None:
            return None
            
        try:
            spectra = self._extract_roi_spectra(combined_mask)
            if spectra is None or spectra.size == 0:
                return None
                
            return ROIStats(
                mean=np.mean(spectra, axis=0),
                std=np.std(spectra, axis=0),
                min=np.min(spectra, axis=0),
                max=np.max(spectra, axis=0),
                count=spectra.shape[0],
                mask=combined_mask
            )
            
        except Exception as e:
            print(f"Error computing combined stats: {e}")
            return None
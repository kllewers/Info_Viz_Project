"""
Utility functions for hyperspectral data processing and application configuration.

Includes normalization, ENVI header parsing helpers, and configuration management.
"""

import numpy as np
import yaml
import os
import json
from typing import Dict, Any, Optional, Tuple, List, Union


def normalize_band_for_display(band_data: np.ndarray, 
                             stretch_percent: float = 2.0,
                             min_val: Optional[float] = None,
                             max_val: Optional[float] = None) -> np.ndarray:
    """
    Normalize band data for display with contrast stretching.
    
    Args:
        band_data: 2D array of band values
        stretch_percent: Percentile for contrast stretching (0-50)
        min_val: Manual minimum value (overrides percentile)
        max_val: Manual maximum value (overrides percentile)
        
    Returns:
        Normalized array (0-255, uint8)
    """
    if band_data.size == 0:
        return np.zeros_like(band_data, dtype=np.uint8)
        
    # Handle NaN and infinite values
    valid_data = band_data[np.isfinite(band_data)]
    
    if valid_data.size == 0:
        return np.zeros_like(band_data, dtype=np.uint8)
        
    # Determine stretch range
    if min_val is None or max_val is None:
        p_low = max(0, stretch_percent)
        p_high = min(100, 100 - stretch_percent)
        
        if min_val is None:
            min_val = np.percentile(valid_data, p_low)
        if max_val is None:
            max_val = np.percentile(valid_data, p_high)
            
    # Avoid division by zero
    if max_val <= min_val:
        max_val = min_val + 1e-6
        
    # Apply stretching
    stretched = np.clip((band_data - min_val) / (max_val - min_val), 0, 1)
    
    # Convert to uint8
    return (stretched * 255).astype(np.uint8)


def create_rgb_composite(red_band: np.ndarray, 
                        green_band: np.ndarray, 
                        blue_band: np.ndarray,
                        stretch_percent: float = 2.0) -> np.ndarray:
    """
    Create RGB composite from three bands.
    
    Args:
        red_band: Red channel data
        green_band: Green channel data  
        blue_band: Blue channel data
        stretch_percent: Percentile for contrast stretching
        
    Returns:
        RGB composite array (height, width, 3)
    """
    # Normalize each band separately
    red_norm = normalize_band_for_display(red_band, stretch_percent)
    green_norm = normalize_band_for_display(green_band, stretch_percent)
    blue_norm = normalize_band_for_display(blue_band, stretch_percent)
    
    # Stack into RGB
    rgb = np.stack([red_norm, green_norm, blue_norm], axis=2)
    
    return rgb


def wavelength_to_band_index(wavelengths: np.ndarray, 
                           target_wavelength: float) -> Tuple[int, float]:
    """
    Find band index closest to target wavelength.
    
    Args:
        wavelengths: Array of wavelength values
        target_wavelength: Target wavelength
        
    Returns:
        Tuple of (band_index, actual_wavelength)
    """
    differences = np.abs(wavelengths - target_wavelength)
    closest_idx = np.argmin(differences)
    
    return int(closest_idx), float(wavelengths[closest_idx])


def find_common_wavelengths(wavelength_sets: List[np.ndarray], 
                          tolerance: float = 1.0) -> np.ndarray:
    """
    Find common wavelengths across multiple datasets.
    
    Args:
        wavelength_sets: List of wavelength arrays
        tolerance: Tolerance for matching wavelengths (nm)
        
    Returns:
        Array of common wavelengths
    """
    if not wavelength_sets:
        return np.array([])
        
    if len(wavelength_sets) == 1:
        return wavelength_sets[0]
        
    # Start with first set
    common = wavelength_sets[0].copy()
    
    # Find wavelengths that exist in all sets
    for wl_set in wavelength_sets[1:]:
        mask = np.zeros(len(common), dtype=bool)
        
        for i, wl in enumerate(common):
            # Check if this wavelength exists in current set
            diffs = np.abs(wl_set - wl)
            if np.min(diffs) <= tolerance:
                mask[i] = True
                
        common = common[mask]
        
    return common


def resample_spectrum(spectrum: np.ndarray, 
                     source_wavelengths: np.ndarray,
                     target_wavelengths: np.ndarray,
                     method: str = 'linear') -> np.ndarray:
    """
    Resample spectrum to new wavelength grid.
    
    Args:
        spectrum: Spectral values
        source_wavelengths: Original wavelengths
        target_wavelengths: Target wavelengths
        method: Interpolation method ('linear', 'nearest', 'cubic')
        
    Returns:
        Resampled spectrum
    """
    from scipy import interpolate
    
    # Remove NaN values
    valid_mask = np.isfinite(spectrum)
    if not np.any(valid_mask):
        return np.full_like(target_wavelengths, np.nan)
        
    valid_spectrum = spectrum[valid_mask]
    valid_wavelengths = source_wavelengths[valid_mask]
    
    # Create interpolation function
    if method == 'linear':
        interp_func = interpolate.interp1d(
            valid_wavelengths, valid_spectrum, 
            kind='linear', fill_value=np.nan, bounds_error=False
        )
    elif method == 'nearest':
        interp_func = interpolate.interp1d(
            valid_wavelengths, valid_spectrum, 
            kind='nearest', fill_value=np.nan, bounds_error=False
        )
    elif method == 'cubic':
        if len(valid_wavelengths) >= 4:
            interp_func = interpolate.interp1d(
                valid_wavelengths, valid_spectrum, 
                kind='cubic', fill_value=np.nan, bounds_error=False
            )
        else:
            # Fall back to linear for small datasets
            interp_func = interpolate.interp1d(
                valid_wavelengths, valid_spectrum, 
                kind='linear', fill_value=np.nan, bounds_error=False
            )
    else:
        raise ValueError(f"Unknown interpolation method: {method}")
        
    return interp_func(target_wavelengths)


def parse_envi_header_value(value_str: str) -> Union[str, List[str], int, float]:
    """
    Parse ENVI header value with proper type conversion.
    
    Args:
        value_str: Raw value string from header
        
    Returns:
        Parsed value with appropriate type
    """
    value_str = value_str.strip()
    
    # Handle multi-line values in braces
    if value_str.startswith('{') and value_str.endswith('}'):
        # Remove braces and split by comma
        inner = value_str.strip('{}')
        values = [v.strip() for v in inner.split(',') if v.strip()]
        
        # Try to convert to numbers
        converted_values = []
        for v in values:
            try:
                # Try integer first
                if '.' not in v and 'e' not in v.lower():
                    converted_values.append(int(v))
                else:
                    converted_values.append(float(v))
            except ValueError:
                converted_values.append(v)
                
        return converted_values
    
    # Single value - try to convert to number
    try:
        # Try integer first
        if '.' not in value_str and 'e' not in value_str.lower():
            return int(value_str)
        else:
            return float(value_str)
    except ValueError:
        return value_str


def estimate_optimal_rgb_bands(wavelengths: np.ndarray) -> Tuple[int, int, int]:
    """
    Estimate optimal RGB band indices based on wavelengths.
    
    Args:
        wavelengths: Array of wavelength values
        
    Returns:
        Tuple of (red_idx, green_idx, blue_idx)
    """
    if wavelengths is None or len(wavelengths) == 0:
        # Fallback when no wavelength info
        n_bands = len(wavelengths) if wavelengths is not None else 100
        return (min(29, n_bands - 1), min(19, n_bands - 1), min(9, n_bands - 1))
    
    # Determine spectral range
    min_wl = np.min(wavelengths)
    max_wl = np.max(wavelengths)
    
    print(f"DEBUG: Wavelength range: {min_wl:.1f} - {max_wl:.1f} nm")
    
    # Choose RGB targets based on spectral range
    if max_wl <= 1000:
        # VNIR range - use standard RGB targets
        target_red = 650    # Red
        target_green = 550  # Green  
        target_blue = 450   # Blue
    elif min_wl >= 1000:
        # SWIR only - use relative distribution in SWIR range
        target_red = max_wl - (max_wl - min_wl) * 0.1   # Near max wavelength
        target_green = min_wl + (max_wl - min_wl) * 0.5  # Middle wavelength
        target_blue = min_wl + (max_wl - min_wl) * 0.1   # Near min wavelength
    else:
        # VNIR-SWIR range - use NIR/SWIR bands
        target_red = 1200   # SWIR
        target_green = 800  # NIR
        target_blue = 650   # Red edge
    
    print(f"DEBUG: RGB targets: R={target_red:.1f}, G={target_green:.1f}, B={target_blue:.1f} nm")
    
    red_idx, actual_red = wavelength_to_band_index(wavelengths, target_red)
    green_idx, actual_green = wavelength_to_band_index(wavelengths, target_green)
    blue_idx, actual_blue = wavelength_to_band_index(wavelengths, target_blue)
    
    # Ensure all bands are different - if they're the same, spread them out
    if red_idx == green_idx == blue_idx:
        n_bands = len(wavelengths)
        red_idx = min(n_bands - 1, n_bands * 3 // 4)      # 75% through bands
        green_idx = min(n_bands - 1, n_bands // 2)        # 50% through bands  
        blue_idx = min(n_bands - 1, n_bands // 4)         # 25% through bands
        print(f"DEBUG: All bands were same, spreading: R={red_idx}, G={green_idx}, B={blue_idx}")
    
    print(f"DEBUG: Selected RGB bands: R={red_idx}({actual_red:.1f}nm), G={green_idx}({actual_green:.1f}nm), B={blue_idx}({actual_blue:.1f}nm)")
    
    return red_idx, green_idx, blue_idx


def get_true_color_rgb_bands(wavelengths: np.ndarray) -> Tuple[int, int, int]:
    """
    Get true color RGB band indices for visible spectrum wavelengths.
    If no visible spectrum bands are available, fall back to well-spaced bands.
    
    Args:
        wavelengths: Array of wavelength values
        
    Returns:
        Tuple of (red_idx, green_idx, blue_idx)
    """
    if wavelengths is None:
        # Fallback when no wavelength info
        return (29, 19, 9)
        
    if len(wavelengths) == 0:
        # Empty array
        return (0, 0, 0)
    
    # Define visible spectrum targets for true color
    true_red = 650      # Red peak around 650nm
    true_green = 550    # Green peak around 550nm
    true_blue = 450     # Blue peak around 450nm
    
    # Check if we have visible spectrum coverage
    min_wl = np.min(wavelengths)
    max_wl = np.max(wavelengths)
    
    # Determine if we have good visible spectrum coverage
    has_blue = min_wl <= 480   # Good blue coverage (need to reach deep blue)
    has_green = min_wl <= 580 and max_wl >= 520  # Good green coverage
    has_red = max_wl >= 620    # Good red coverage
    
    if has_blue and has_green and has_red:
        # We have good visible spectrum coverage - use true color bands
        red_idx, _ = wavelength_to_band_index(wavelengths, true_red)
        green_idx, _ = wavelength_to_band_index(wavelengths, true_green)
        blue_idx, _ = wavelength_to_band_index(wavelengths, true_blue)
        
        print(f"DEBUG: True color RGB selected - good VIS coverage")
        return red_idx, green_idx, blue_idx
    else:
        # Limited visible spectrum - use well-spaced bands across available range
        n_bands = len(wavelengths)
        
        # Ensure we have at least 3 different bands
        if n_bands < 3:
            return 0, 0, 0
            
        # Space bands evenly across the spectral range
        # Use bands from high, middle, and low wavelength regions
        red_idx = min(n_bands - 1, int(n_bands * 0.8))    # 80% through (longest wavelengths)
        green_idx = min(n_bands - 1, int(n_bands * 0.5))  # 50% through (middle wavelengths)
        blue_idx = min(n_bands - 1, int(n_bands * 0.2))   # 20% through (shortest wavelengths)
        
        # Make sure all bands are different
        if red_idx == green_idx:
            if red_idx > 0:
                green_idx = red_idx - 1
            else:
                green_idx = min(n_bands - 1, red_idx + 1)
                
        if blue_idx == green_idx:
            if blue_idx > 1:
                blue_idx = green_idx - 2
            else:
                blue_idx = min(n_bands - 1, green_idx + 2)
                
        if blue_idx == red_idx:
            if blue_idx > 2:
                blue_idx = red_idx - 3
            else:
                blue_idx = min(n_bands - 1, red_idx + 3)
        
        print(f"DEBUG: Fallback RGB selected - limited VIS coverage, using well-spaced bands")
        print(f"DEBUG: Selected bands: R={red_idx}({wavelengths[red_idx]:.1f}nm), "
              f"G={green_idx}({wavelengths[green_idx]:.1f}nm), "
              f"B={blue_idx}({wavelengths[blue_idx]:.1f}nm)")
        
        return red_idx, green_idx, blue_idx


class ConfigManager:
    """Configuration management for hyperspectral viewer."""
    
    def __init__(self, config_file: Optional[str] = None):
        self.config_file = config_file or 'configs/default.yaml'
        self.config = self._load_default_config()
        
        # Load user config if exists
        if os.path.exists(self.config_file):
            self.load_config(self.config_file)
            
    def _load_default_config(self) -> Dict[str, Any]:
        """Load default configuration."""
        return {
            'display': {
                'default_rgb_bands': [29, 19, 9],
                'contrast_stretch_percent': 2.0,
                'auto_scale': True,
                'background_color': 'white'
            },
            'data_loading': {
                'default_load_mode': 'memmap',  # 'memmap' or 'ram'
                'memmap_threshold_mb': 1000
            },
            'roi': {
                'default_roi_color': 'red',
                'roi_line_width': 2,
                'max_rois': 50
            },
            'spectrum_plot': {
                'line_width': 2,
                'grid_alpha': 0.3,
                'legend_position': 'top-right'
            },
            'export': {
                'default_format': 'csv',
                'include_wavelengths': True,
                'precision': 6
            },
            'ui': {
                'window_width': 1200,
                'window_height': 800,
                'splitter_ratios': [0.6, 0.4],
                'toolbar_icon_size': 24
            }
        }
        
    def load_config(self, filename: str) -> bool:
        """Load configuration from file."""
        try:
            if filename.endswith(('.yaml', '.yml')):
                with open(filename, 'r') as f:
                    user_config = yaml.safe_load(f)
            elif filename.endswith('.json'):
                with open(filename, 'r') as f:
                    user_config = json.load(f)
            else:
                print(f"Unsupported config format: {filename}")
                return False
                
            # Merge with defaults
            self._merge_config(self.config, user_config)
            return True
            
        except Exception as e:
            print(f"Error loading config {filename}: {e}")
            return False
            
    def save_config(self, filename: Optional[str] = None) -> bool:
        """Save configuration to file."""
        try:
            save_file = filename or self.config_file
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(save_file), exist_ok=True)
            
            if save_file.endswith(('.yaml', '.yml')):
                with open(save_file, 'w') as f:
                    yaml.dump(self.config, f, default_flow_style=False, indent=2)
            elif save_file.endswith('.json'):
                with open(save_file, 'w') as f:
                    json.dump(self.config, f, indent=2)
            else:
                print(f"Unsupported config format: {save_file}")
                return False
                
            return True
            
        except Exception as e:
            print(f"Error saving config: {e}")
            return False
            
    def _merge_config(self, default: Dict[str, Any], user: Dict[str, Any]):
        """Recursively merge user config with defaults."""
        for key, value in user.items():
            if key in default and isinstance(default[key], dict) and isinstance(value, dict):
                self._merge_config(default[key], value)
            else:
                default[key] = value
                
    def get(self, key_path: str, default_value: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        
        try:
            for key in keys:
                value = value[key]
            return value
        except (KeyError, TypeError):
            return default_value
            
    def set(self, key_path: str, value: Any):
        """Set configuration value using dot notation."""
        keys = key_path.split('.')
        config_section = self.config
        
        # Navigate to parent section
        for key in keys[:-1]:
            if key not in config_section:
                config_section[key] = {}
            config_section = config_section[key]
            
        # Set value
        config_section[keys[-1]] = value


def validate_envi_file_pair(data_file: str) -> Tuple[bool, Optional[str], Optional[str]]:
    """
    Validate ENVI data file and find corresponding header.
    
    Args:
        data_file: Path to ENVI data file (with or without extension)
        
    Returns:
        Tuple of (is_valid, data_path, header_path)
    """
    if not os.path.exists(data_file):
        return False, None, None
        
    # Find header file - enhanced for extensionless files
    possible_headers = [
        data_file + '.hdr',  # Standard: datafile.hdr
        os.path.splitext(data_file)[0] + '.hdr',  # Remove extension, add .hdr
        data_file.replace('.bsq', '.hdr').replace('.bil', '.hdr').replace('.bip', '.hdr')
    ]
    
    # For extensionless files, check if data_file itself might be the data
    # and look for corresponding .hdr files
    if '.' not in os.path.basename(data_file):
        # Extensionless file - could be raw ENVI data
        possible_headers.extend([
            data_file + '.hdr',  # filename.hdr
            os.path.dirname(data_file) + '/' + os.path.basename(data_file) + '.hdr'  # same dir
        ])
    
    header_file = None
    for hdr in possible_headers:
        if os.path.exists(hdr):
            header_file = hdr
            break
            
    if header_file is None:
        return False, data_file, None
        
    return True, data_file, header_file


def format_memory_size(size_bytes: int) -> str:
    """Format memory size in human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def estimate_memory_usage(shape: Tuple[int, int, int], dtype: np.dtype) -> int:
    """Estimate memory usage for hyperspectral data."""
    total_elements = np.prod(shape)
    bytes_per_element = dtype.itemsize
    return total_elements * bytes_per_element
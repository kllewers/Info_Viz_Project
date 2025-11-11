"""
DataHandler class for ENVI-format hyperspectral data loading and management.

Uses spectral.io.envi for proper handling of BSQ/BIL/BIP formats with correct dimensions.
"""

import numpy as np
import os
from typing import Optional, Tuple, Dict, Any
from pathlib import Path
import warnings

try:
    import spectral.io.envi as envi
    SPECTRAL_AVAILABLE = True
except ImportError:
    SPECTRAL_AVAILABLE = False
    print("Warning: spectral package not available. Install with: pip install spectral")

try:
    import netCDF4 as nc
    import xarray as xr
    NETCDF_AVAILABLE = True
except ImportError:
    NETCDF_AVAILABLE = False
    print("Warning: netCDF4/xarray packages not available. Install with: pip install netCDF4 xarray")


class DataHandler:
    """Handles ENVI and EMIT format hyperspectral cubes with lazy loading and efficient access."""
    
    def __init__(self):
        self.spy_file = None  # SpyFile object from spectral.io (ENVI files)
        self.nc_dataset = None  # NetCDF dataset (EMIT files)
        self.data = None  # Memmap or loaded array
        self.header = {}
        self.wavelengths = None
        self.shape = None  # (rows/lines, cols/samples, bands) - spectral.io standard
        self.interleave = None
        self.data_type = None
        self.filename = None
        self.header_filename = None
        self.is_loaded = False
        self.use_memmap = True
        self.file_type = None  # 'envi' or 'emit'
        self.fill_value = -9999.0  # EMIT fill value
        self.bad_band_list = None  # BBL - array of 0s and 1s (0=bad, 1=good)
        self.data_ignore_value = None  # Data ignore value for no-data pixels
        
    def load_envi_data(self, filename: str, load_to_ram: bool = False, force_interleave: str = None) -> bool:
        """
        Load ENVI or EMIT format hyperspectral data.
        
        Args:
            filename: Path to ENVI data file (.bsq, .bil, .bip, .hdr) or EMIT file (.nc)
            load_to_ram: If True, load entire dataset to RAM; if False, use memmap
            force_interleave: Force specific interleave format ('BSQ', 'BIL', 'BIP') for ENVI files
            
        Returns:
            True if successful, False otherwise
        """
        # Detect file type
        if self._is_emit_file(filename):
            return self._load_emit_data(filename, load_to_ram)
        elif self._is_aviris3_file(filename):
            return self._load_aviris3_data(filename, load_to_ram)
        else:
            return self._load_envi_data_original(filename, load_to_ram, force_interleave)
    
    def _is_emit_file(self, filename: str) -> bool:
        """Check if file is an EMIT NetCDF file."""
        path = Path(filename)
        return (path.suffix.lower() == '.nc' and
                'EMIT' in path.name.upper())

    def _is_aviris3_file(self, filename: str) -> bool:
        """Check if file is an AVIRIS-3 NetCDF file."""
        path = Path(filename)
        return (path.suffix.lower() == '.nc' and
                'AV3' in path.name.upper())
    
    def _load_envi_data_original(self, filename: str, load_to_ram: bool = False, force_interleave: str = None) -> bool:
        """Load ENVI format hyperspectral data using spectral.io."""
        if not SPECTRAL_AVAILABLE:
            print("Error: spectral package not available. Install with: pip install spectral")
            return False
            
        try:
            # Determine if input is header file or data file
            if filename.endswith('.hdr'):
                header_file = filename
                data_file = self._find_data_file(header_file)
            else:
                data_file = filename
                header_file = self._find_header_file(filename)
                
            if not header_file or not os.path.exists(header_file):
                print(f"Error: Could not find header file for {filename}")
                return False
                
            if not data_file or not os.path.exists(data_file):
                print(f"Error: Could not find data file for {filename}")
                return False
            
            # Open using spectral.io.envi with optional interleave override
            if force_interleave:
                print(f"Forcing interleave format: {force_interleave}")
                self.spy_file = self._create_spy_file_with_interleave(header_file, data_file, force_interleave)
            else:
                self.spy_file = envi.open(header_file, data_file)
            
            # Store filenames and file type
            self.filename = data_file
            self.header_filename = header_file
            self.file_type = 'envi'
            
            # Extract metadata from SpyFile
            self.header = dict(self.spy_file.metadata)
            self.shape = self.spy_file.shape  # (rows, cols, bands) in spectral.io
            self.interleave = self.spy_file.interleave
            self.data_type = self.spy_file.dtype
            
            # Get wavelengths - spectral.io handles this automatically
            if hasattr(self.spy_file, 'bands') and hasattr(self.spy_file.bands, 'centers'):
                self.wavelengths = np.array(self.spy_file.bands.centers)
            else:
                # Fallback to band numbers if no wavelength info
                self.wavelengths = np.arange(1, self.shape[2] + 1)
                
            # Load bad band list (BBL) from header if present
            self._load_bad_band_list_from_header()
            
            # Load data ignore value from header if present
            self._load_data_ignore_value_from_header()
                
            self.use_memmap = not load_to_ram
            
            # Set up data access
            if load_to_ram:
                print(f"Loading {self.shape} dataset to RAM...")
                self.data = self.spy_file.load()  # Load to RAM as ImageArray
            else:
                # Use SpyFile's memmap functionality
                self.data = self.spy_file.open_memmap(writable=False)
                print(f"Created memmap for {self.shape} dataset")
                
            self.is_loaded = True
            print(f"Successfully loaded ENVI data: {self.shape} ({self.interleave})")
            return True
            
        except Exception as e:
            print(f"Error loading ENVI data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _load_emit_data(self, filename: str, load_to_ram: bool = False) -> bool:
        """Load EMIT NetCDF format hyperspectral data with correct dimension handling."""
        if not NETCDF_AVAILABLE:
            print("Error: netCDF4/xarray packages not available. Install with: pip install netCDF4 xarray")
            return False
            
        try:
            file_path = Path(filename)
            if not file_path.exists():
                print(f"Error: EMIT file not found: {filename}")
                return False
                
            print(f"Loading EMIT file: {filename}")
            
            # First, analyze the NetCDF file structure to understand dimensions
            main_var_name = None
            main_var_dims = None
            main_var_shape = None
            
            try:
                with nc.Dataset(filename, 'r') as nc_file:
                    print(f"NetCDF groups: {list(nc_file.groups.keys())}")
                    print(f"Root variables: {list(nc_file.variables.keys())}")
                    print(f"Root dimensions: {dict(nc_file.dimensions)}")
                    
                    # Find the main 3D data variable with proper dimension analysis
                    candidates = []
                    for var_name, var_obj in nc_file.variables.items():
                        if len(var_obj.dimensions) == 3:
                            candidates.append({
                                'name': var_name,
                                'dimensions': var_obj.dimensions,
                                'shape': var_obj.shape
                            })
                            print(f"Found 3D variable '{var_name}': dims={var_obj.dimensions}, shape={var_obj.shape}")
                    
                    # Prioritize standard EMIT variable names
                    for candidate in candidates:
                        if candidate['name'] in ['reflectance', 'radiance', 'mask']:
                            main_var_name = candidate['name']
                            main_var_dims = candidate['dimensions']
                            main_var_shape = candidate['shape']
                            break
                    
                    # If no standard names found, use first 3D variable
                    if main_var_name is None and candidates:
                        main_var_name = candidates[0]['name']
                        main_var_dims = candidates[0]['dimensions']
                        main_var_shape = candidates[0]['shape']
                        
                    if main_var_name is None:
                        print("Error: No suitable 3D data variable found in EMIT file")
                        return False
                    
                    print(f"Selected main variable: '{main_var_name}' with dimensions {main_var_dims} = {main_var_shape}")
                    
            except Exception as e:
                print(f"Error inspecting NetCDF structure: {e}")
                return False
            
            # Determine product level from filename
            product_level = self._get_emit_product_level(file_path.name)
            print(f"Detected EMIT product level: {product_level}")
            
            # Determine correct transpose based on dimension analysis
            transpose_order = self._determine_emit_transpose(main_var_dims, main_var_shape)
            print(f"Determined transpose order: {transpose_order}")
            
            if load_to_ram:
                # Load with xarray for full dataset loading
                with xr.open_dataset(filename, decode_cf=True) as ds:
                    print(f"XArray data variables: {list(ds.data_vars.keys())}")
                    
                    if main_var_name not in ds.data_vars:
                        print(f"Error: Selected variable '{main_var_name}' not found in xarray dataset")
                        return False
                    
                    print(f"Using data variable: {main_var_name}")
                    
                    # Extract data array with correct dimension handling
                    data_array = ds[main_var_name].values
                    print(f"Raw data array shape: {data_array.shape}")
                    
                    # Get wavelengths from sensor_band_parameters group
                    try:
                        with xr.open_dataset(filename, group='sensor_band_parameters') as sensor_params:
                            wavelengths = sensor_params['wavelengths'].values
                            print(f"Loaded wavelengths from sensor_band_parameters: {wavelengths.shape}")
                    except Exception as wl_error:
                        print(f"Could not load wavelengths: {wl_error}")
                        # Fallback: create wavelengths based on spectral dimension
                        spectral_size = self._get_spectral_dimension_size(main_var_shape, main_var_dims)
                        wavelengths = np.arange(1, spectral_size + 1)
                        print(f"Using fallback wavelengths: {wavelengths.shape}")
                        
                # Apply transpose to get (rows, cols, bands) for DOODA compatibility
                if transpose_order is not None:
                    data_array = np.transpose(data_array, transpose_order)
                    print(f"Transposed data array shape: {data_array.shape}")
                
            else:
                # Use netCDF4 for memory mapping approach
                self.nc_dataset = nc.Dataset(filename, 'r')
                print(f"NetCDF4 variables: {list(self.nc_dataset.variables.keys())}")
                
                if main_var_name not in self.nc_dataset.variables:
                    print(f"Error: Selected variable '{main_var_name}' not found in NetCDF dataset")
                    return False
                
                print(f"Using data variable: {main_var_name}")
                
                reflectance_var = self.nc_dataset.variables[main_var_name]
                print(f"Variable dimensions: {reflectance_var.dimensions}")
                print(f"Variable shape: {reflectance_var.shape}")
                
                # Load data array with correct dimension handling
                data_array = np.array(reflectance_var)
                print(f"Raw data array shape: {data_array.shape}")
                
                # Apply transpose to get (rows, cols, bands) for DOODA compatibility
                if transpose_order is not None:
                    data_array = np.transpose(data_array, transpose_order)
                    print(f"Transposed data array shape: {data_array.shape}")
                
                # Extract wavelengths with proper error handling
                try:
                    if 'sensor_band_parameters' in self.nc_dataset.groups:
                        sensor_group = self.nc_dataset.groups['sensor_band_parameters']
                        if 'wavelengths' in sensor_group.variables:
                            wavelengths = np.array(sensor_group.variables['wavelengths'])
                            print(f"Loaded wavelengths from sensor_band_parameters: {wavelengths.shape}")
                        else:
                            raise KeyError("wavelengths variable not found in sensor_band_parameters group")
                    else:
                        raise KeyError("sensor_band_parameters group not found")
                except Exception as wl_error:
                    print(f"Could not extract wavelengths: {wl_error}")
                    # Fallback: create wavelengths based on final spectral dimension
                    wavelengths = np.arange(1, data_array.shape[2] + 1)
                    print(f"Using fallback wavelengths: {wavelengths.shape}")
                
            # Handle fill values and create mask
            data_array = self._handle_emit_fill_values(data_array)
            
            # Store results in DataHandler format
            self.data = data_array
            self.wavelengths = wavelengths
            self.filename = str(file_path)
            self.file_type = 'emit'
            self.shape = data_array.shape  # (rows, cols, bands)
            self.interleave = 'bsq'  # EMIT data organization
            self.data_type = data_array.dtype
            self.use_memmap = not load_to_ram
            
            # Create metadata compatible with existing interface
            self.header = {
                'lines': self.shape[0],
                'samples': self.shape[1], 
                'bands': self.shape[2],
                'data_type': self._numpy_to_envi_dtype(data_array.dtype),
                'interleave': 'bsq',
                'wavelength_units': 'nm',
                'sensor_type': 'EMIT',
                'product_level': product_level,
                'spatial_resolution': '60m'
            }
            
            # Extract spatial metadata if available
            try:
                self._extract_emit_spatial_metadata()
            except Exception as e:
                print(f"Warning: Could not extract spatial metadata: {e}")
                
            self.is_loaded = True
            print(f"Successfully loaded EMIT data: {self.shape}")
            return True
            
        except Exception as e:
            print(f"Error loading EMIT data: {e}")
            import traceback
            traceback.print_exc()
            return False

    def _load_aviris3_data(self, filename: str, load_to_ram: bool = False) -> bool:
        """Load AVIRIS-3 NetCDF format hyperspectral data with correct dimension handling."""
        if not NETCDF_AVAILABLE:
            print("Error: netCDF4/xarray packages not available. Install with: pip install netCDF4 xarray")
            return False

        try:
            file_path = Path(filename)
            if not file_path.exists():
                print(f"Error: AVIRIS-3 file not found: {filename}")
                return False

            print(f"Loading AVIRIS-3 file: {filename}")

            # Analyze the NetCDF file structure
            main_var_name = None
            main_var_dims = None
            main_var_shape = None

            try:
                with nc.Dataset(filename, 'r') as nc_file:
                    print(f"NetCDF groups: {list(nc_file.groups.keys())}")
                    print(f"Root variables: {list(nc_file.variables.keys())}")
                    print(f"Root dimensions: {dict(nc_file.dimensions)}")

                    # AVIRIS-3 can have either reflectance (L2A) or radiance (L1B) data
                    data_group_name = None
                    data_var_name = None

                    if 'reflectance' in nc_file.groups:
                        data_group_name = 'reflectance'
                        data_var_name = 'reflectance'
                    elif 'radiance' in nc_file.groups:
                        data_group_name = 'radiance'
                        data_var_name = 'radiance'
                    else:
                        print("Error: No reflectance or radiance group found in AVIRIS-3 file")
                        return False

                    data_group = nc_file.groups[data_group_name]
                    if data_var_name in data_group.variables:
                        var_obj = data_group.variables[data_var_name]
                        main_var_name = f'{data_group_name}/{data_var_name}'
                        main_var_dims = var_obj.dimensions
                        main_var_shape = var_obj.shape
                        print(f"Found AVIRIS-3 {data_var_name} variable: dims={main_var_dims}, shape={main_var_shape}")
                    else:
                        print(f"Error: No {data_var_name} variable found in {data_group_name} group")
                        return False

            except Exception as e:
                print(f"Error inspecting NetCDF structure: {e}")
                return False

            # Determine product level from filename
            product_level = self._get_aviris3_product_level(file_path.name)
            print(f"Detected AVIRIS-3 product level: {product_level}")

            # Determine correct transpose - AVIRIS-3 uses (wavelength, northing/lines, easting/samples)
            # We need to convert to (rows, cols, bands) for DOODA compatibility
            transpose_order = (1, 2, 0)  # Move wavelength from first to last dimension
            print(f"Determined transpose order: {transpose_order}")

            if load_to_ram:
                # Load with xarray for full dataset loading
                with xr.open_dataset(filename, group=data_group_name) as ds:
                    print(f"XArray {data_group_name} group variables: {list(ds.data_vars.keys())}")

                    if data_var_name not in ds.data_vars:
                        print(f"Error: {data_var_name} variable not found in xarray dataset")
                        return False

                    # Extract data array with correct dimension handling
                    data_array = ds[data_var_name].values
                    print(f"Raw data array shape: {data_array.shape}")

                    # Get wavelengths from the same group
                    if 'wavelength' in ds.variables:
                        wavelengths = ds['wavelength'].values
                        print(f"Loaded wavelengths from {data_group_name} group: {wavelengths.shape}")
                    else:
                        print(f"Warning: No wavelength variable found in {data_group_name} group")
                        wavelengths = np.arange(1, data_array.shape[0] + 1)

            else:
                # Use netCDF4 for memory mapping approach
                self.nc_dataset = nc.Dataset(filename, 'r')
                data_group = self.nc_dataset.groups[data_group_name]

                data_var = data_group.variables[data_var_name]
                print(f"Variable dimensions: {data_var.dimensions}")
                print(f"Variable shape: {data_var.shape}")

                # Load data array
                data_array = np.array(data_var)
                print(f"Raw data array shape: {data_array.shape}")

                # Extract wavelengths
                if 'wavelength' in data_group.variables:
                    wavelengths = np.array(data_group.variables['wavelength'])
                    print(f"Loaded wavelengths from {data_group_name} group: {wavelengths.shape}")
                else:
                    print(f"Warning: No wavelength variable found in {data_group_name} group")
                    wavelengths = np.arange(1, data_array.shape[0] + 1)

            # Apply transpose to get (rows, cols, bands) for DOODA compatibility
            if transpose_order is not None:
                data_array = np.transpose(data_array, transpose_order)
                print(f"Transposed data array shape: {data_array.shape}")

            # Handle fill values (AVIRIS-3 uses NaN for invalid pixels and -9999 fill values)
            data_array = self._handle_aviris3_fill_values(data_array, data_var_name)

            # Store results in DataHandler format
            self.data = data_array
            self.wavelengths = wavelengths
            self.filename = str(file_path)
            self.file_type = 'aviris3'
            self.shape = data_array.shape  # (rows, cols, bands)
            self.interleave = 'bsq'  # AVIRIS-3 data organization after transpose
            self.data_type = data_array.dtype
            self.use_memmap = not load_to_ram

            # Create metadata compatible with existing interface
            self.header = {
                'lines': self.shape[0],
                'samples': self.shape[1],
                'bands': self.shape[2],
                'data_type': self._numpy_to_envi_dtype(data_array.dtype),
                'interleave': 'bsq',
                'wavelength_units': 'nm',
                'sensor_type': 'AVIRIS-3',
                'product_level': product_level,
                'spatial_resolution': 'variable'  # AVIRIS-3 resolution depends on flight altitude
            }

            # Extract spatial metadata if available
            try:
                self._extract_aviris3_spatial_metadata()
            except Exception as e:
                print(f"Warning: Could not extract spatial metadata: {e}")

            self.is_loaded = True
            print(f"Successfully loaded AVIRIS-3 data: {self.shape}")
            return True

        except Exception as e:
            print(f"Error loading AVIRIS-3 data: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _get_emit_product_level(self, filename: str) -> str:
        """Extract product level from EMIT filename."""
        if 'L1B' in filename:
            return 'L1B_RAD'
        elif 'L2A_RFL' in filename:
            return 'L2A_RFL'
        elif 'L2A_MASK' in filename:
            return 'L2A_MASK'
        elif 'L2B_MIN' in filename:
            return 'L2B_MIN'
        else:
            return 'Unknown'

    def _get_aviris3_product_level(self, filename: str) -> str:
        """Extract product level from AVIRIS-3 filename."""
        if 'L1B_RDN' in filename:
            return 'L1B_RDN'  # Calibrated Radiance
        elif 'L2A_RFL' in filename:
            return 'L2A_RFL'  # Surface Reflectance
        elif 'L2A_OE' in filename:
            return 'L2A_OE'   # Optimal Estimation (includes reflectance)
        elif 'L2B' in filename:
            return 'L2B'      # Higher level products
        else:
            return 'Unknown'
    
    def _determine_emit_transpose(self, dimensions: tuple, shape: tuple) -> tuple:
        """
        Determine the correct transpose order for EMIT data to convert to (rows, cols, bands).
        
        Args:
            dimensions: NetCDF dimension names tuple (e.g., ('bands', 'downtrack', 'crosstrack'))
            shape: Data shape tuple (e.g., (285, 1242, 1242))
            
        Returns:
            Transpose order tuple or None if no transpose needed
        """
        # EMIT standard dimension names and expected patterns
        spectral_names = ['bands', 'wavelength', 'spectral']
        spatial_names = ['downtrack', 'crosstrack', 'x', 'y', 'lat', 'lon']
        
        dim_types = []
        for i, (dim_name, dim_size) in enumerate(zip(dimensions, shape)):
            dim_name_lower = dim_name.lower()
            
            # Classify dimension by name and size
            if any(spec in dim_name_lower for spec in spectral_names) or (200 <= dim_size <= 400):
                dim_types.append(('spectral', i, dim_name, dim_size))
            elif any(spatial in dim_name_lower for spatial in spatial_names) or dim_size >= 50:
                dim_types.append(('spatial', i, dim_name, dim_size))
            else:
                dim_types.append(('unknown', i, dim_name, dim_size))
        
        print(f"Dimension classification: {dim_types}")
        
        # Find spectral dimension index
        spectral_dims = [dt for dt in dim_types if dt[0] == 'spectral']
        if not spectral_dims:
            print("Warning: No spectral dimension identified, assuming first dimension is spectral")
            spectral_index = 0
        else:
            spectral_index = spectral_dims[0][1]
        
        # EMIT data is typically (bands, downtrack, crosstrack)
        # We want to convert to (downtrack, crosstrack, bands) = (rows, cols, bands)
        if spectral_index == 0:
            # Spectral dimension is first, move to last
            return (1, 2, 0)
        elif spectral_index == 2:
            # Spectral dimension is already last
            return None
        else:
            # Spectral dimension is in middle (unusual for EMIT)
            print(f"Warning: Unusual spectral dimension position: {spectral_index}")
            if len(shape) == 3:
                # Create transpose to move spectral to last position
                other_indices = [i for i in range(3) if i != spectral_index]
                return tuple(other_indices + [spectral_index])
            
        return None
    
    def _get_spectral_dimension_size(self, shape: tuple, dimensions: tuple) -> int:
        """
        Get the size of the spectral dimension.
        
        Args:
            shape: Data shape tuple
            dimensions: NetCDF dimension names tuple
            
        Returns:
            Size of spectral dimension
        """
        transpose_order = self._determine_emit_transpose(dimensions, shape)
        
        if transpose_order is None:
            # No transpose needed, assume last dimension is spectral
            return shape[-1]
        elif transpose_order == (1, 2, 0):
            # First dimension becomes last after transpose
            return shape[0]
        else:
            # Find which original dimension becomes the spectral (last) dimension
            spectral_original_index = transpose_order.index(len(transpose_order) - 1)
            return shape[spectral_original_index]
    
    def _handle_emit_fill_values(self, data_array: np.ndarray) -> np.ndarray:
        """Handle EMIT fill values and invalid data."""
        # Mask fill values
        fill_mask = (data_array == self.fill_value)

        # For reflectance data, mask values outside valid range (0-1)
        if self.file_type == 'emit':
            valid_range_mask = (data_array < 0) | (data_array > 1)
            data_array = np.ma.masked_where(fill_mask | valid_range_mask, data_array)
        else:
            data_array = np.ma.masked_where(fill_mask, data_array)

        return data_array

    def _handle_aviris3_fill_values(self, data_array: np.ndarray, data_type: str = 'reflectance') -> np.ndarray:
        """Handle AVIRIS-3 fill values and invalid data.

        Args:
            data_array: Input data array
            data_type: Either 'reflectance' or 'radiance' to apply appropriate range checking
        """
        # AVIRIS-3 uses NaN for invalid pixels and -9999 as fill value
        nan_mask = np.isnan(data_array)
        fill_value_mask = data_array == -9999.0

        if data_type == 'reflectance':
            # For reflectance data, mask values outside valid range (0-1)
            # AVIRIS-3 reflectance should be in range 0-1 for surface reflectance
            valid_range_mask = (data_array < 0) | (data_array > 1)
        else:
            # For radiance data, only mask negative values (radiance should be >= 0)
            # Don't apply upper limit as radiance can have varying ranges
            valid_range_mask = data_array < 0

        # Create masked array
        data_array = np.ma.masked_where(nan_mask | fill_value_mask | valid_range_mask, data_array)

        return data_array
    
    def _extract_emit_spatial_metadata(self):
        """Extract spatial reference information from EMIT dataset."""
        if self.nc_dataset is None:
            return
            
        try:
            location_group = self.nc_dataset.groups['location']
            
            # Get basic spatial info without loading full coordinate arrays
            lat_var = location_group.variables['lat']
            lon_var = location_group.variables['lon']
            
            # Add spatial info to header
            self.header.update({
                'coordinate_system': 'WGS84 Geographic (EPSG:4326)',
                'spatial_dimensions': lat_var.shape,
                'spatial_units': 'degrees'
            })
            
        except Exception as e:
            print(f"Warning: Could not extract spatial metadata: {e}")

    def _extract_aviris3_spatial_metadata(self):
        """Extract spatial reference information from AVIRIS-3 dataset."""
        if self.nc_dataset is None:
            return

        try:
            # AVIRIS-3 has root-level spatial coordinate variables
            if 'easting' in self.nc_dataset.variables and 'northing' in self.nc_dataset.variables:
                easting_var = self.nc_dataset.variables['easting']
                northing_var = self.nc_dataset.variables['northing']

                # Add spatial info to header
                self.header.update({
                    'coordinate_system': 'UTM (projected)',
                    'easting_range': (float(easting_var[0]), float(easting_var[-1])),
                    'northing_range': (float(northing_var[0]), float(northing_var[-1])),
                    'spatial_units': 'm'  # UTM coordinates are in meters
                })

            # Check for coordinate reference system information
            if 'transverse_mercator' in self.nc_dataset.variables:
                crs_var = self.nc_dataset.variables['transverse_mercator']
                if hasattr(crs_var, 'spatial_ref'):
                    self.header['spatial_reference'] = crs_var.spatial_ref

        except Exception as e:
            print(f"Warning: Could not extract AVIRIS-3 spatial metadata: {e}")
    
    def _numpy_to_envi_dtype(self, numpy_dtype) -> int:
        """Convert numpy dtype to ENVI data type code."""
        dtype_map = {
            np.uint8: 1, np.int16: 2, np.int32: 3, np.float32: 4,
            np.float64: 5, np.complex64: 6, np.complex128: 9,
            np.uint16: 12, np.uint32: 13, np.int64: 14, np.uint64: 15
        }
        return dtype_map.get(numpy_dtype.type, 4)  # Default to float32
    
    def _create_spy_file_with_interleave(self, header_file: str, data_file: str, interleave: str):
        """Create SpyFile with specific interleave format by temporarily modifying header."""
        try:
            import tempfile
            import shutil
            
            # Read original header
            with open(header_file, 'r') as f:
                header_content = f.read()
            
            # Create temporary header with forced interleave
            temp_header = None
            try:
                # Create temporary header file
                temp_header = tempfile.NamedTemporaryFile(mode='w', suffix='.hdr', delete=False)
                
                # Modify interleave line in header content
                lines = header_content.split('\n')
                modified_lines = []
                found_interleave = False
                
                for line in lines:
                    if line.strip().lower().startswith('interleave'):
                        # Replace interleave line
                        modified_lines.append(f'interleave = {interleave.lower()}')
                        found_interleave = True
                        print(f"Modified interleave line: interleave = {interleave.lower()}")
                    else:
                        modified_lines.append(line)
                
                # If no interleave line found, add it
                if not found_interleave:
                    # Find a good place to insert it (after dimensions)
                    insert_idx = len(modified_lines)
                    for i, line in enumerate(modified_lines):
                        if any(dim in line.lower() for dim in ['bands', 'lines', 'samples']):
                            insert_idx = max(insert_idx, i + 1)
                    modified_lines.insert(insert_idx, f'interleave = {interleave.lower()}')
                    print(f"Added interleave line: interleave = {interleave.lower()}")
                
                # Write modified header
                temp_header.write('\n'.join(modified_lines))
                temp_header.close()
                
                print(f"Created temporary header with forced {interleave} interleave: {temp_header.name}")
                
                # Use spectral.io.envi.open with modified header
                spy_file = envi.open(temp_header.name, data_file)
                
                print(f"Successfully created {interleave} SpyFile: {spy_file.shape}, interleave: {spy_file.interleave}")
                return spy_file
                
            finally:
                # Clean up temporary file
                if temp_header and os.path.exists(temp_header.name):
                    os.unlink(temp_header.name)
                    
        except Exception as e:
            print(f"Error creating SpyFile with interleave {interleave}: {e}")
            import traceback
            traceback.print_exc()
            print("Falling back to auto-detection...")
            return envi.open(header_file, data_file)
    
    def _load_bad_band_list_from_header(self):
        """Load bad band list (BBL) from ENVI header."""
        if not self.header:
            return
            
        try:
            # Check for BBL in header (case insensitive)
            bbl_key = None
            for key in self.header.keys():
                if key.lower() == 'bbl':
                    bbl_key = key
                    break
                    
            if bbl_key and self.header[bbl_key]:
                bbl_data = self.header[bbl_key]
                
                # Handle different BBL formats from header
                if isinstance(bbl_data, str):
                    # Parse string format like "{1, 1, 0, 1, 1}"
                    bbl_str = bbl_data.strip('{}[] ')
                    if bbl_str:
                        bbl_values = [int(x.strip()) for x in bbl_str.split(',') if x.strip()]
                        self.bad_band_list = np.array(bbl_values, dtype=np.int32)
                elif isinstance(bbl_data, (list, tuple)):
                    # Direct list/tuple format
                    self.bad_band_list = np.array(bbl_data, dtype=np.int32)
                elif hasattr(bbl_data, '__iter__'):
                    # Array-like object
                    self.bad_band_list = np.array(list(bbl_data), dtype=np.int32)
                    
                # Validate BBL length matches number of bands
                if self.bad_band_list is not None and len(self.bad_band_list) != self.shape[2]:
                    print(f"Warning: BBL length ({len(self.bad_band_list)}) doesn't match number of bands ({self.shape[2]})")
                    self.bad_band_list = None
                else:
                    print(f"Loaded BBL from header: {np.sum(self.bad_band_list == 0)} bad bands out of {len(self.bad_band_list)}")
                    
        except Exception as e:
            print(f"Warning: Error loading bad band list from header: {e}")
            self.bad_band_list = None
    
    def _load_data_ignore_value_from_header(self):
        """Load data ignore value from ENVI header."""
        if not self.header:
            return
            
        try:
            # Check for data ignore value in header (case insensitive)
            ignore_key = None
            target_keys = ['data ignore value', 'data_ignore_value', 'ignore_value']
            for key in self.header.keys():
                if key.lower() in target_keys:
                    ignore_key = key
                    break
                    
            if ignore_key and self.header[ignore_key] is not None:
                ignore_value = self.header[ignore_key]
                
                # Convert to appropriate numeric type
                if isinstance(ignore_value, str):
                    ignore_value = ignore_value.strip()
                    if ignore_value.lower() in ['none', 'null', '']:
                        self.data_ignore_value = None
                    else:
                        self.data_ignore_value = float(ignore_value)
                else:
                    self.data_ignore_value = float(ignore_value)
                    
                if self.data_ignore_value is not None:
                    print(f"Loaded data ignore value from header: {self.data_ignore_value}")
                    
        except Exception as e:
            print(f"Warning: Error loading data ignore value from header: {e}")
            self.data_ignore_value = None
    
    def set_bad_band_list(self, bad_band_list: np.ndarray) -> bool:
        """
        Set bad band list and save to header file.
        
        Args:
            bad_band_list: Array of 0s and 1s where 0=bad band, 1=good band
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded or self.file_type != 'envi':
            print("Error: Can only set bad band list for loaded ENVI files")
            return False
            
        if bad_band_list is None:
            self.bad_band_list = None
        else:
            bad_band_list = np.array(bad_band_list, dtype=np.int32)
            
            # Validate length matches number of bands
            if len(bad_band_list) != self.shape[2]:
                print(f"Error: BBL length ({len(bad_band_list)}) must match number of bands ({self.shape[2]})")
                return False
                
            # Validate values are 0 or 1
            if not np.all((bad_band_list == 0) | (bad_band_list == 1)):
                print("Error: BBL values must be 0 (bad) or 1 (good)")
                return False
                
            self.bad_band_list = bad_band_list
        
        # Save to header file
        return self._save_bad_band_list_to_header()
    
    def set_data_ignore_value(self, ignore_value: float) -> bool:
        """
        Set data ignore value and save to header file.
        
        Args:
            ignore_value: Value to ignore as no-data, or None to remove
            
        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded or self.file_type != 'envi':
            print("Error: Can only set data ignore value for loaded ENVI files")
            return False
            
        self.data_ignore_value = ignore_value
        
        # Save to header file
        return self._save_data_ignore_value_to_header()
    
    def _save_bad_band_list_to_header(self) -> bool:
        """Save bad band list (BBL) to ENVI header file."""
        if not self.header_filename:
            print("Error: No header filename available for saving BBL")
            return False
            
        if not os.path.exists(self.header_filename):
            print(f"Error: Header file does not exist: {self.header_filename}")
            return False
            
        # Check file permissions
        if not os.access(self.header_filename, os.W_OK):
            print(f"Error: Header file is not writable: {self.header_filename}")
            return False
            
        try:
            # Read current header content
            with open(self.header_filename, 'r') as f:
                header_lines = f.readlines()
            
            # Remove existing BBL line if present
            filtered_lines = []
            for line in header_lines:
                if not line.strip().lower().startswith('bbl'):
                    filtered_lines.append(line)
            
            # Add new BBL line if we have bad band list
            if self.bad_band_list is not None:
                bbl_str = '{' + ', '.join(map(str, self.bad_band_list)) + '}'
                bbl_line = f'bbl = {bbl_str}\n'
                
                # Insert after wavelength-related lines or at end
                insert_idx = len(filtered_lines)
                for i, line in enumerate(filtered_lines):
                    line_lower = line.strip().lower()
                    if line_lower.startswith(('wavelength', 'fwhm', 'bands')):
                        insert_idx = max(insert_idx, i + 1)
                
                filtered_lines.insert(insert_idx, bbl_line)
                print(f"Added BBL to header: {np.sum(self.bad_band_list == 0)} bad bands out of {len(self.bad_band_list)}")
            else:
                print("Removed BBL from header")
            
            # Write updated header
            with open(self.header_filename, 'w') as f:
                f.writelines(filtered_lines)
                
            # Update internal header dictionary
            if self.bad_band_list is not None:
                self.header['bbl'] = self.bad_band_list.tolist()
            elif 'bbl' in self.header:
                del self.header['bbl']
                
            return True
            
        except Exception as e:
            print(f"Error saving bad band list to header: {e}")
            return False
    
    def _save_data_ignore_value_to_header(self) -> bool:
        """Save data ignore value to ENVI header file."""
        if not self.header_filename or not os.path.exists(self.header_filename):
            print("Error: No header file available for saving data ignore value")
            return False
            
        try:
            # Read current header content
            with open(self.header_filename, 'r') as f:
                header_lines = f.readlines()
            
            # Remove existing data ignore value line if present
            filtered_lines = []
            for line in header_lines:
                line_lower = line.strip().lower()
                if not line_lower.startswith(('data ignore value', 'data_ignore_value')):
                    filtered_lines.append(line)
            
            # Add new data ignore value line if we have one
            if self.data_ignore_value is not None:
                ignore_line = f'data ignore value = {self.data_ignore_value}\n'
                
                # Insert after data type or bands lines
                insert_idx = len(filtered_lines)
                for i, line in enumerate(filtered_lines):
                    line_lower = line.strip().lower()
                    if line_lower.startswith(('data type', 'bands', 'lines', 'samples')):
                        insert_idx = max(insert_idx, i + 1)
                
                filtered_lines.insert(insert_idx, ignore_line)
                print(f"Added data ignore value to header: {self.data_ignore_value}")
            else:
                print("Removed data ignore value from header")
            
            # Write updated header
            with open(self.header_filename, 'w') as f:
                f.writelines(filtered_lines)
                
            # Update internal header dictionary
            if self.data_ignore_value is not None:
                self.header['data ignore value'] = self.data_ignore_value
            else:
                for key in list(self.header.keys()):
                    if key.lower() in ['data ignore value', 'data_ignore_value', 'ignore_value']:
                        del self.header[key]
                        break
                
            return True
            
        except Exception as e:
            print(f"Error saving data ignore value to header: {e}")
            return False
    
    def reload_from_header(self) -> bool:
        """
        Reload the file to pick up changes from the header file.
        This is useful after updating BBL or data ignore values.
        
        Returns:
            True if successful, False otherwise
        """
        if not self.is_loaded or self.file_type != 'envi':
            print("Error: Can only reload ENVI files")
            return False
            
        if not self.filename:
            print("Error: No filename available for reload")
            return False
            
        # Store current settings
        current_load_to_ram = not self.use_memmap
        
        # Reload the file
        print(f"Reloading {self.filename} to pick up header changes...")
        success = self.load_envi_data(self.filename, load_to_ram=current_load_to_ram)
        
        if success:
            print("File reloaded successfully with updated header information")
        else:
            print("Error: Failed to reload file")
            
        return success
    
    def get_bad_band_list(self) -> Optional[np.ndarray]:
        """
        Get the current bad band list.
        
        Returns:
            Array of 0s and 1s where 0=bad band, 1=good band, or None if not set
        """
        return self.bad_band_list
    
    def get_good_bands(self) -> Optional[np.ndarray]:
        """
        Get indices of good bands (where BBL=1).
        
        Returns:
            Array of good band indices, or None if BBL not set
        """
        if self.bad_band_list is None:
            return None
        return np.where(self.bad_band_list == 1)[0]
    
    def get_bad_bands(self) -> Optional[np.ndarray]:
        """
        Get indices of bad bands (where BBL=0).
        
        Returns:
            Array of bad band indices, or None if BBL not set
        """
        if self.bad_band_list is None:
            return None
        return np.where(self.bad_band_list == 0)[0]
    
    def get_data_ignore_value(self) -> Optional[float]:
        """
        Get the current data ignore value.
        
        Returns:
            Data ignore value, or None if not set
        """
        return self.data_ignore_value
    
    def is_band_good(self, band_index: int) -> bool:
        """
        Check if a specific band is good (not in bad band list).
        
        Args:
            band_index: Band index (0-based)
            
        Returns:
            True if band is good or BBL not set, False if band is bad
        """
        if self.bad_band_list is None:
            return True  # Assume all bands are good if BBL not set
        if band_index < 0 or band_index >= len(self.bad_band_list):
            return True  # Invalid index, assume good
        return self.bad_band_list[band_index] == 1
    
    def _find_header_file(self, filename: str) -> Optional[str]:
        """Find the corresponding ENVI header file."""
        possible_headers = [
            filename + '.hdr',
            os.path.splitext(filename)[0] + '.hdr',
            filename.replace('.bsq', '.hdr').replace('.bil', '.hdr').replace('.bip', '.hdr')
        ]
        
        for header_file in possible_headers:
            if os.path.exists(header_file):
                return header_file
        return None
    
    def _find_data_file(self, header_file: str) -> Optional[str]:
        """Find the corresponding ENVI data file from header file."""
        base_name = os.path.splitext(header_file)[0]
        
        # Common ENVI data file extensions
        possible_data_files = [
            base_name,  # No extension
            base_name + '.bsq',
            base_name + '.bil', 
            base_name + '.bip',
            base_name + '.dat',
            base_name + '.img'
        ]
        
        for data_file in possible_data_files:
            if os.path.exists(data_file):
                return data_file
        return None
    
    def get_band_data(self, band_index: int) -> Optional[np.ndarray]:
        """
        Get data for a specific band.
        
        Args:
            band_index: Band index (0-based)
            
        Returns:
            2D array of band data (rows, cols) or None
        """
        if not self.is_loaded:
            return None
            
        if band_index < 0 or band_index >= self.shape[2]:
            return None
            
        try:
            if self.use_memmap and self.data is not None:
                return self.data[:, :, band_index]
            else:
                return self.spy_file.read_band(band_index)
        except Exception as e:
            print(f"Error getting band {band_index}: {e}")
            return None
    
    def get_pixel_spectrum(self, x: int, y: int) -> Optional[np.ndarray]:
        """
        Get spectrum for a specific pixel.
        
        Args:
            x: Column index (sample) - 0-based
            y: Row index (line) - 0-based
            
        Returns:
            Spectrum array or None if invalid coordinates
        """
        if not self.is_loaded:
            return None
            
        # Validate coordinates against shape (rows, cols, bands)
        if x < 0 or x >= self.shape[1] or y < 0 or y >= self.shape[0]:
            return None
            
        try:
            if self.file_type == 'emit':
                # For EMIT data, use direct indexing on transposed array
                spectrum = self.data[y, x, :]
                # Handle masked values for EMIT data
                if np.ma.is_masked(spectrum):
                    spectrum = spectrum.filled(np.nan)
                return np.array(spectrum, dtype=np.float64)

            elif self.file_type == 'aviris3':
                # For AVIRIS-3 data, use direct indexing on transposed array
                spectrum = self.data[y, x, :]
                # Handle masked values for AVIRIS-3 data
                if np.ma.is_masked(spectrum):
                    spectrum = spectrum.filled(np.nan)
                return np.array(spectrum, dtype=np.float64)

            elif self.file_type == 'envi':
                if self.use_memmap and self.data is not None:
                    # For ENVI memmap, use direct indexing
                    spectrum = self.data[y, x, :]
                else:
                    # For loaded ENVI data, use SpyFile's read_pixel method
                    spectrum = self.spy_file.read_pixel(y, x)
                    
                return np.array(spectrum, dtype=np.float64)
            else:
                print(f"Error: Unknown file type: {self.file_type}")
                return None
                
        except Exception as e:
            print(f"Error getting pixel spectrum at ({x}, {y}): {e}")
            return None
    
    def extract_line_spectra(self, line_index: int) -> Optional[np.ndarray]:
        """
        Extract spectral data for all spatial positions along a single line.
        
        Args:
            line_index: Row index (spatial line) - 0-based
            
        Returns:
            Array of shape (spatial_positions, bands) containing spectral data
            for all positions along the specified line, or None if invalid line
        """
        if not self.is_loaded:
            return None
            
        # Validate line index against shape (rows, cols, bands)
        if line_index < 0 or line_index >= self.shape[0]:
            return None
            
        try:
            if self.file_type == 'emit':
                # For EMIT data, extract entire line at once
                line_spectra = self.data[line_index, :, :]  # Shape: (cols, bands)
                # Handle masked values for EMIT data
                if np.ma.is_masked(line_spectra):
                    line_spectra = line_spectra.filled(np.nan)
                return np.array(line_spectra, dtype=np.float64)

            elif self.file_type == 'aviris3':
                # For AVIRIS-3 data, extract entire line at once
                line_spectra = self.data[line_index, :, :]  # Shape: (cols, bands)
                # Handle masked values for AVIRIS-3 data
                if np.ma.is_masked(line_spectra):
                    line_spectra = line_spectra.filled(np.nan)
                return np.array(line_spectra, dtype=np.float64)

            elif self.file_type == 'envi':
                if self.use_memmap and self.data is not None:
                    # For ENVI memmap, extract line directly
                    line_spectra = self.data[line_index, :, :]  # Shape: (cols, bands)
                else:
                    # For loaded ENVI data, read line pixel by pixel
                    cols, bands = self.shape[1], self.shape[2]
                    line_spectra = np.zeros((cols, bands), dtype=np.float64)
                    
                    for col in range(cols):
                        try:
                            spectrum = self.spy_file.read_pixel(line_index, col)
                            line_spectra[col, :] = spectrum
                        except Exception as pixel_error:
                            print(f"Warning: Error reading pixel ({line_index}, {col}): {pixel_error}")
                            line_spectra[col, :] = np.nan
                    
                return line_spectra
            else:
                print(f"Error: Unknown file type: {self.file_type}")
                return None
                
        except Exception as e:
            print(f"Error extracting line spectra at line {line_index}: {e}")
            return None
    
    def get_rgb_composite(self, red_band: int = None, green_band: int = None, blue_band: int = None,
                        stretch_percent: float = 2.0, no_data_value: float = None) -> Optional[np.ndarray]:
        """
        Create RGB composite image from specified bands.
        
        Args:
            red_band: Band index for red channel (0-based), None for auto-select
            green_band: Band index for green channel (0-based), None for auto-select 
            blue_band: Band index for blue channel (0-based), None for auto-select
            stretch_percent: Percentile for contrast stretching (0-50), default 2.0
            no_data_value: Value to exclude from contrast stretching, None to use loaded data ignore value
            
        Returns:
            RGB image array (rows, cols, 3) or None
        """
        if not self.is_loaded:
            return None
            
        # Use loaded data ignore value if no_data_value not explicitly provided
        if no_data_value is None and self.data_ignore_value is not None:
            no_data_value = self.data_ignore_value
            
        try:
            # Auto-select RGB bands if not specified
            if red_band is None or green_band is None or blue_band is None:
                rgb_bands = self._estimate_rgb_bands()
                red_band = red_band if red_band is not None else rgb_bands[0]
                green_band = green_band if green_band is not None else rgb_bands[1]
                blue_band = blue_band if blue_band is not None else rgb_bands[2]
            
            # Validate band indices
            max_bands = self.shape[2]
            red_band = min(max(red_band, 0), max_bands - 1)
            green_band = min(max(green_band, 0), max_bands - 1)
            blue_band = min(max(blue_band, 0), max_bands - 1)
            
            if self.file_type == 'emit':
                # For EMIT data, use direct indexing on transposed array
                red = self.data[:, :, red_band]
                green = self.data[:, :, green_band]
                blue = self.data[:, :, blue_band]

                # Handle masked arrays for EMIT data
                if np.ma.is_masked(red):
                    red = red.filled(0)
                if np.ma.is_masked(green):
                    green = green.filled(0)
                if np.ma.is_masked(blue):
                    blue = blue.filled(0)

            elif self.file_type == 'aviris3':
                # For AVIRIS-3 data, use direct indexing on transposed array
                red = self.data[:, :, red_band]
                green = self.data[:, :, green_band]
                blue = self.data[:, :, blue_band]

                # Handle masked arrays for AVIRIS-3 data
                if np.ma.is_masked(red):
                    red = red.filled(0)
                if np.ma.is_masked(green):
                    green = green.filled(0)
                if np.ma.is_masked(blue):
                    blue = blue.filled(0)

            elif self.file_type == 'envi':
                if self.use_memmap and self.data is not None:
                    # For ENVI memmap: data shape is always (rows, cols, bands)
                    red = self.data[:, :, red_band]
                    green = self.data[:, :, green_band]  
                    blue = self.data[:, :, blue_band]
                else:
                    # Use SpyFile's read_band method for reliable band extraction
                    red = self.spy_file.read_band(red_band)
                    green = self.spy_file.read_band(green_band)
                    blue = self.spy_file.read_band(blue_band)
            else:
                print(f"Error: Unknown file type: {self.file_type}")
                return None
            
            # Check if mono mode (all bands are the same)
            if red_band == green_band == blue_band:
                # Return grayscale for monochromatic mode
                return self._normalize_for_display(red[:, :, np.newaxis], stretch_percent, no_data_value)[:, :, 0]
            else:
                # Stack RGB channels for normal RGB mode
                rgb = np.stack([red, green, blue], axis=2)
                return self._normalize_for_display(rgb, stretch_percent, no_data_value)
            
        except Exception as e:
            print(f"Error creating RGB composite: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def _estimate_rgb_bands(self) -> Tuple[int, int, int]:
        """Estimate good RGB band indices based on wavelengths."""
        if self.wavelengths is None or len(self.wavelengths) == 0:
            # Fallback for cases without wavelength info
            n_bands = self.shape[2]
            return (min(29, n_bands - 1), min(19, n_bands - 1), min(9, n_bands - 1))
            
        # Target wavelengths for RGB (nm)
        target_r = 650  # Red
        target_g = 550  # Green  
        target_b = 450  # Blue
        
        # Find closest bands to target wavelengths
        r_band = np.argmin(np.abs(self.wavelengths - target_r))
        g_band = np.argmin(np.abs(self.wavelengths - target_g))
        b_band = np.argmin(np.abs(self.wavelengths - target_b))
        
        return (int(r_band), int(g_band), int(b_band))
    
    def _normalize_for_display(self, image: np.ndarray, stretch_percent: float = 2.0, no_data_value: float = None) -> np.ndarray:
        """Normalize image data for display (0-255) with percentile-based stretching.
        
        Args:
            image: Input image array
            stretch_percent: Percentile for contrast stretching
            no_data_value: Value to exclude from contrast stretching calculations
        """
        # Handle each band separately
        normalized = np.zeros_like(image, dtype=np.uint8)
        
        # Debugging removed - normalization working correctly
        
        for i in range(image.shape[2]):
            band = image[:, :, i].astype(np.float64)
            
            # Remove NaN, infinite values, and no data values
            valid_mask = np.isfinite(band)
            
            # Additional filtering for no data value
            if no_data_value is not None:
                if np.isnan(no_data_value):
                    # No data value is NaN - already handled by isfinite
                    pass
                else:
                    # Filter out the specific no data value with tolerance for floating point comparison
                    # Use a more appropriate tolerance for large values
                    tolerance = max(abs(no_data_value) * 1e-6, 1e-6) if no_data_value != 0 else 1e-6
                    no_data_mask = ~np.isclose(band, no_data_value, atol=tolerance, rtol=1e-6)
                    valid_mask = valid_mask & no_data_mask
            
            valid_data = band[valid_mask]
            if valid_data.size == 0:
                # Band has no valid data
                normalized[:, :, i] = 0
                continue
                
            # Band statistics look good
            
            # Use specified percentiles for contrast stretching
            try:
                p_low = stretch_percent
                p_high = 100 - stretch_percent
                p2, p98 = np.percentile(valid_data, [p_low, p_high])
                # Using percentile stretch
            except:
                # Fallback to min/max if percentile fails
                p2, p98 = np.min(valid_data), np.max(valid_data)
                # Fallback to min/max stretch
            
            if p98 > p2:
                # Apply linear stretch to all pixels first
                band_stretched = (band - p2) / (p98 - p2)
                band_clipped = np.clip(band_stretched, 0, 1)
                normalized[:, :, i] = (band_clipped * 255).astype(np.uint8)
            else:
                # Constant value case
                if p2 != 0:
                    normalized[:, :, i] = 128  # Mid-gray for constant non-zero values
                else:
                    normalized[:, :, i] = 0
            
            # Set no data pixels to black (0) after stretching
            if no_data_value is not None and not np.isnan(no_data_value):
                # Use the same tolerance as in the contrast stretching calculation
                tolerance = max(abs(no_data_value) * 1e-6, 1e-6) if no_data_value != 0 else 1e-6
                no_data_mask = np.isclose(band, no_data_value, atol=tolerance, rtol=1e-6)
                normalized[no_data_mask, i] = 0  # Set no data pixels to black
                    
        # Normalization complete
        return normalized
    
    def get_band_by_wavelength(self, target_wavelength: float) -> Optional[Tuple[int, float]]:
        """
        Find the band index closest to a target wavelength.
        
        Args:
            target_wavelength: Target wavelength value
            
        Returns:
            Tuple of (band_index, actual_wavelength) or None
        """
        if self.wavelengths is None:
            return None
            
        differences = np.abs(self.wavelengths - target_wavelength)
        closest_idx = np.argmin(differences)
        
        return int(closest_idx), float(self.wavelengths[closest_idx])
    
    def get_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        if not self.is_loaded:
            return {}
            
        info = {
            'filename': self.filename,
            'file_format': self.file_type,
            'shape': self.shape,  # (rows, cols, bands)
            'interleave': self.interleave,
            'use_memmap': self.use_memmap,
            'wavelength_range': (float(self.wavelengths[0]), float(self.wavelengths[-1])) if self.wavelengths is not None and len(self.wavelengths) > 0 else None,
            'num_bands': self.shape[2],
            'spatial_size': (self.shape[0], self.shape[1]),  # (rows, cols)
            'dtype': str(self.data_type)
        }
        
        # Add file-type specific metadata
        if self.file_type == 'envi':
            info['header_filename'] = self.header_filename
            if self.spy_file:
                info['envi_file_type'] = type(self.spy_file).__name__
                if hasattr(self.spy_file, 'scale_factor'):
                    info['scale_factor'] = self.spy_file.scale_factor
                if hasattr(self.spy_file, 'offset'):
                    info['offset'] = self.spy_file.offset
        elif self.file_type == 'emit':
            info['product_level'] = self.header.get('product_level', 'Unknown')
            info['sensor_type'] = 'EMIT'
            info['spatial_resolution'] = self.header.get('spatial_resolution', '60m')
            info['coordinate_system'] = self.header.get('coordinate_system', 'WGS84')
        elif self.file_type == 'aviris3':
            info['product_level'] = self.header.get('product_level', 'Unknown')
            info['sensor_type'] = 'AVIRIS-3'
            info['spatial_resolution'] = self.header.get('spatial_resolution', 'variable')
            info['coordinate_system'] = self.header.get('coordinate_system', 'UTM')
                
        return info
    
    def cleanup(self):
        """Clean up resources, especially for EMIT NetCDF files."""
        if self.nc_dataset is not None:
            try:
                self.nc_dataset.close()
                self.nc_dataset = None
            except Exception as e:
                print(f"Warning: Error closing NetCDF dataset: {e}")
        
        # Reset other attributes
        self.is_loaded = False
        self.data = None
        self.spy_file = None
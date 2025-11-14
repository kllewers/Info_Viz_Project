import numpy as np

def identify_channels(wavelengths, ranges):
    """
    Identify the channel indices that correspond to the specified wavelength ranges,
    with automatic unit normalization (nm ↔ μm).

    Parameters:
    wavelengths (list or dict): List of wavelengths (floats) or dict of {band: wavelength}.
    ranges (dict): Dictionary defining wavelength ranges, e.g.,
                   {'Red': (600, 700)} in nm or {'Red': (0.6, 0.7)} in μm.

    Returns:
    dict: Dictionary with range names as keys and lists of channel indices as values.
    """
    # Handle dictionary vs list
    if isinstance(wavelengths, dict):
        band_items = sorted(wavelengths.items())  # list of (band, wavelength)
    else:
        band_items = [(i + 1, wl) for i, wl in enumerate(wavelengths)]  # rasterio is 1-indexed

    # Unit normalization: if max wavelength < 10 in either input, assume μm and convert both to nm
    if max(wl for _, wl in band_items) < 10 and max(max(r) for r in ranges.values()) > 10:
        print("Detected μm in wavelengths and nm in ranges — converting wavelengths to nm...")
        band_items = [(band, wl * 1000) for band, wl in band_items]
    elif max(max(r) for r in ranges.values()) < 10 and max(wl for _, wl in band_items) > 100:
        print("Detected nm in wavelengths and μm in ranges — converting ranges to nm...")
        ranges = {k: (low * 1000, high * 1000) for k, (low, high) in ranges.items()}

    # Channel matching
    channels = {key: [] for key in ranges}
    for band, wl in band_items:
        for color_range, (low, high) in ranges.items():
            if low <= wl <= high:
                channels[color_range].append(band)
                print(f"Channel {band} with wavelength {wl:.2f} nm falls in {color_range} range.")

    return channels

def print_channel_summary(channels):
    for color_range, indices in channels.items():
        print(f"Channels for {color_range}: {indices}")

def average_bands_by_range(dataset, channel_dict):
    """
    Average the bands in each spectral range and return as a dictionary of arrays.

    Parameters:
    dataset (rasterio.DatasetReader): Opened rasterio dataset.
    channel_dict (dict): Output from identify_channels(), mapping ranges to band indices.

    Returns:
    dict: A dictionary with keys as range names and values as 2D numpy arrays of averaged bands.
    """
    averaged = {}
    for label, bands in channel_dict.items():
        if not bands:
            print(f"No bands found for range '{label}' — skipping.")
            continue

        print(f"Averaging bands {bands} for range '{label}'...")
        stack = np.stack([dataset.read(band) for band in bands])
        averaged[label] = np.mean(stack, axis=0)

    return averaged




#-------------------------------------
# Spectral Indices Utility functions
#-------------------------------------

#-------------------------------------
# 1. Broadband Greenness
#-------------------------------------

# NDVI
def calculate_ndvi(red_band, nir_band):
    """
    Calculate NDVI from Red and NIR bands.

    Parameters:
    red_band (np.ndarray): 2D array of Red reflectance values.
    nir_band (np.ndarray): 2D array of NIR reflectance values.

    Returns:
    np.ndarray: 2D NDVI array.
    """
    np.seterr(divide='ignore', invalid='ignore')  # Handle divide-by-zero gracefully
    ndvi = (nir_band - red_band) / (nir_band + red_band)
    ndvi = np.clip(ndvi, -1.0, 1.0)  # Optional: clip to valid NDVI range
    return ndvi

# Simple Ratio Index (SR)
def calculate_sr(red_band, nir_band):
    """
    Calculate the Simple Ratio Index (SR) from Red and NIR bands.

    SR = NIR / Red

    Parameters:
    red_band (np.ndarray): 2D array of Red reflectance values.
    nir_band (np.ndarray): 2D array of NIR reflectance values.

    Returns:
    np.ndarray: 2D SR array.
    """
    np.seterr(divide='ignore', invalid='ignore')  # Handle divide-by-zero gracefully
    sr = nir_band / red_band
    sr = np.nan_to_num(sr, nan=0.0, posinf=30.0, neginf=0.0)  # Replace NaNs and infs with bounds
    return sr

# Enhanced Vegetation Index (EVI)
def calculate_evi(red_band, nir_band, blue_band):
    """
    Calculate the Enhanced Vegetation Index (EVI) from Red, NIR, and Blue bands.

    EVI = 2.5 * ((NIR - RED) / (NIR + 6*RED - 7.5*BLUE + 1))

    Parameters:
    red_band (np.ndarray): 2D array of Red reflectance values.
    nir_band (np.ndarray): 2D array of NIR reflectance values.
    blue_band (np.ndarray): 2D array of Blue reflectance values.

    Returns:
    np.ndarray: 2D EVI array.
    """
    np.seterr(divide='ignore', invalid='ignore')  # Handle divide-by-zero gracefully

    numerator = nir_band - red_band
    denominator = nir_band + 6 * red_band - 7.5 * blue_band + 1
    evi = 2.5 * (numerator / denominator)

    evi = np.clip(evi, -1.0, 1.0)  # Optional: clip to valid EVI range
    return evi

# Atmospherically Resistant Vegetation Index (ARVI): 3
import numpy as np

def calculate_arvi(red_band, nir_band, blue_band):
    """
    Calculate the Atmospherically Resistant Vegetation Index (ARVI).

    ARVI = (NIR - (2 * RED - BLUE)) / (NIR + (2 * RED - BLUE))

    Parameters:
    red_band (np.ndarray): 2D array of Red reflectance values.
    nir_band (np.ndarray): 2D array of NIR reflectance values.
    blue_band (np.ndarray): 2D array of Blue reflectance values.

    Returns:
    np.ndarray: 2D ARVI array.
    """
    np.seterr(divide='ignore', invalid='ignore')  # Handle divide-by-zero gracefully

    corrected_red = 2 * red_band - blue_band
    numerator = nir_band - corrected_red
    denominator = nir_band + corrected_red

    arvi = numerator / denominator
    arvi = np.clip(arvi, -1.0, 1.0)  # Clip to typical ARVI range
    return arvi

# Sum Green Index (SG): 
def calculate_sg(reflectance_cube, wavelengths):
    """
    Calculate the Sum Green Index (SG) across the 500–600 nm range.

    SG is the mean reflectance across bands in the 500–600 nm range.

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D SG array (mean reflectance across green wavelengths).
    """
    # Convert to numpy array if needed
    wavelengths = np.array(wavelengths)

    # Identify indices for wavelengths between 500 and 600 nm
    green_band_indices = np.where((wavelengths >= 500) & (wavelengths <= 600))[0]

    if green_band_indices.size == 0:
        raise ValueError("No bands found in the 500–600 nm range.")

    # Extract those bands and compute mean reflectance
    green_reflectance = reflectance_cube[green_band_indices, :, :]
    sg = np.mean(green_reflectance, axis=0)

    return sg

#-------------------------------------
# 2. Narrowband Greenness
#-------------------------------------
# Red Edge Normalized Difference Vegetation Index (NDVI705): 4
def calculate_ndvi705(reflectance_cube, wavelengths):
    """
    Calculate the Red Edge Normalized Difference Vegetation Index (NDVI705).

    NDVI705 = (R750 - R705) / (R750 + R705)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D NDVI705 array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices of the bands closest to 705 nm and 750 nm
    idx_705 = np.argmin(np.abs(wavelengths - 705))
    idx_750 = np.argmin(np.abs(wavelengths - 750))

    # Extract the corresponding bands
    band_705 = reflectance_cube[idx_705, :, :]
    band_750 = reflectance_cube[idx_750, :, :]

    # Calculate NDVI705
    np.seterr(divide='ignore', invalid='ignore')
    ndvi705 = (band_750 - band_705) / (band_750 + band_705)
    ndvi705 = np.clip(ndvi705, -1.0, 1.0)

    return ndvi705


# Modified Red Edge Simple Ratio Index (mSR705): 4
def calculate_sg(reflectance_cube, wavelengths):
    """
    Calculate the Sum Green Index (SG) across the 500–600 nm range.

    SG is the mean reflectance across bands in the 500–600 nm range.

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D SG array (mean reflectance across green wavelengths).
    """
    # Convert to numpy array if needed
    wavelengths = np.array(wavelengths)

    # Identify indices for wavelengths between 500 and 600 nm
    green_band_indices = np.where((wavelengths >= 500) & (wavelengths <= 600))[0]

    if green_band_indices.size == 0:
        raise ValueError("No bands found in the 500–600 nm range.")

    # Extract those bands and compute mean reflectance
    green_reflectance = reflectance_cube[green_band_indices, :, :]
    sg = np.mean(green_reflectance, axis=0)

    return sg

# Modified Red Edge Normalized Difference Vegetation Index (mNDVI705) 5


# Vogelmann Red Edge Index 1 (VOG1): 5
def calculate_vog1(reflectance_cube, wavelengths):
    """
    Calculate the Vogelmann Red Edge Index 1 (VOG1).

    VOG1 = R740 / R720

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D VOG1 array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 740nm and 720nm bands
    idx_740 = np.argmin(np.abs(wavelengths - 740))
    idx_720 = np.argmin(np.abs(wavelengths - 720))

    # Extract reflectance bands
    band_740 = reflectance_cube[idx_740, :, :]
    band_720 = reflectance_cube[idx_720, :, :]

    # Calculate VOG1
    np.seterr(divide='ignore', invalid='ignore')
    vog1 = band_740 / band_720
    vog1 = np.nan_to_num(vog1, nan=0.0, posinf=20.0, neginf=0.0)

    return vog1


# Vogelmann Red Edge Index 2 (VOG2): 5
def calculate_vog2(reflectance_cube, wavelengths):
    """
    Calculate the Vogelmann Red Edge Index 2 (VOG2).

    VOG2 = (R734 - R747) / (R715 + R726)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D VOG2 array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for required bands
    idx_715 = np.argmin(np.abs(wavelengths - 715))
    idx_726 = np.argmin(np.abs(wavelengths - 726))
    idx_734 = np.argmin(np.abs(wavelengths - 734))
    idx_747 = np.argmin(np.abs(wavelengths - 747))

    # Extract corresponding bands
    band_715 = reflectance_cube[idx_715, :, :]
    band_726 = reflectance_cube[idx_726, :, :]
    band_734 = reflectance_cube[idx_734, :, :]
    band_747 = reflectance_cube[idx_747, :, :]

    # Calculate VOG2
    np.seterr(divide='ignore', invalid='ignore')
    numerator = band_734 - band_747
    denominator = band_715 + band_726
    vog2 = numerator / denominator
    vog2 = np.nan_to_num(vog2, nan=0.0, posinf=20.0, neginf=0.0)

    return vog2


# Vogelmann Red Edge Index 3 (VOG3): 5
import numpy as np

def calculate_vog3(reflectance_cube, wavelengths):
    """
    Calculate the Vogelmann Red Edge Index 3 (VOG3).

    VOG3 = (R734 - R747) / (R715 + R720)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D VOG3 array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for required wavelengths
    idx_715 = np.argmin(np.abs(wavelengths - 715))
    idx_720 = np.argmin(np.abs(wavelengths - 720))
    idx_734 = np.argmin(np.abs(wavelengths - 734))
    idx_747 = np.argmin(np.abs(wavelengths - 747))

    # Extract reflectance bands
    band_715 = reflectance_cube[idx_715, :, :]
    band_720 = reflectance_cube[idx_720, :, :]
    band_734 = reflectance_cube[idx_734, :, :]
    band_747 = reflectance_cube[idx_747, :, :]

    # Calculate VOG3
    np.seterr(divide='ignore', invalid='ignore')
    numerator = band_734 - band_747
    denominator = band_715 + band_720
    vog3 = numerator / denominator
    vog3 = np.nan_to_num(vog3, nan=0.0, posinf=20.0, neginf=0.0)

    return vog3


# Red Edge Position Index (REP): 5
def calculate_rep(reflectance_cube, wavelengths):
    """
    Calculate the Red Edge Position Index (REP) using the first derivative method.

    REP = wavelength of maximum first derivative of reflectance in the 690–740 nm range.

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D REP array, with values in nanometers indicating red edge inflection point.
    """
    wavelengths = np.array(wavelengths)
    
    # Get band indices in the 690–740 nm range
    red_edge_range = np.where((wavelengths >= 690) & (wavelengths <= 740))[0]
    
    if red_edge_range.size < 3:
        raise ValueError("Not enough bands in the 690–740 nm range to compute REP.")

    # Extract the subset of the reflectance cube and corresponding wavelengths
    refl_subset = reflectance_cube[red_edge_range, :, :]  # Shape: (bands, rows, cols)
    wl_subset = wavelengths[red_edge_range]

    # Calculate first derivative along the wavelength axis (axis=0)
    dR = np.gradient(refl_subset, wl_subset, axis=0)  # dR/dλ

    # Find the index of max derivative for each pixel
    max_deriv_indices = np.argmax(dR, axis=0)  # Shape: (rows, cols)

    # Map index of max derivative back to wavelength values
    rep = wl_subset[max_deriv_indices]

    return rep


#----------------------------------------
# Light Use Efficiency 6
#-----------------------------------------

# Photochemical Reflectance Index (PRI): 6
def calculate_pri(reflectance_cube, wavelengths):
    """
    Calculate the Photochemical Reflectance Index (PRI).

    PRI = (R531 - R570) / (R531 + R570)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D PRI array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 531 nm and 570 nm bands
    idx_531 = np.argmin(np.abs(wavelengths - 531))
    idx_570 = np.argmin(np.abs(wavelengths - 570))

    # Extract reflectance bands
    band_531 = reflectance_cube[idx_531, :, :]
    band_570 = reflectance_cube[idx_570, :, :]

    # Calculate PRI
    np.seterr(divide='ignore', invalid='ignore')
    pri = (band_531 - band_570) / (band_531 + band_570)
    pri = np.clip(pri, -1.0, 1.0)

    return pri

# Structure Insensitive Pigment Index (SIPI): 6
def calculate_sipi(reflectance_cube, wavelengths):
    """
    Calculate the Structure Insensitive Pigment Index (SIPI).

    SIPI = (R800 - R445) / (R800 - R680)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D SIPI array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 445nm, 680nm, and 800nm bands
    idx_445 = np.argmin(np.abs(wavelengths - 445))
    idx_680 = np.argmin(np.abs(wavelengths - 680))
    idx_800 = np.argmin(np.abs(wavelengths - 800))

    # Extract corresponding reflectance bands
    band_445 = reflectance_cube[idx_445, :, :]
    band_680 = reflectance_cube[idx_680, :, :]
    band_800 = reflectance_cube[idx_800, :, :]

    # Calculate SIPI
    np.seterr(divide='ignore', invalid='ignore')
    sipi = (band_800 - band_445) / (band_800 - band_680)
    sipi = np.clip(sipi, 0.0, 2.0)  # Optional: clip to expected range

    return sipi


# Red Green Ratio Index (RGR Ratio): 7
def calculate_rgr_ratio(reflectance_cube, wavelengths):
    """
    Calculate the Red Green Ratio (RGR Ratio).

    RGR = mean(RED) / mean(GREEN)
    RED range: ~600–700 nm
    GREEN range: ~500–600 nm

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D RGR Ratio array.
    """
    wavelengths = np.array(wavelengths)

    # Identify indices for red and green bands
    red_indices = np.where((wavelengths >= 600) & (wavelengths <= 700))[0]
    green_indices = np.where((wavelengths >= 500) & (wavelengths <= 600))[0]

    if red_indices.size == 0 or green_indices.size == 0:
        raise ValueError("Insufficient red or green wavelength coverage in input data.")

    # Compute mean reflectance across red and green bands
    mean_red = np.mean(reflectance_cube[red_indices, :, :], axis=0)
    mean_green = np.mean(reflectance_cube[green_indices, :, :], axis=0)

    # Calculate RGR Ratio
    np.seterr(divide='ignore', invalid='ignore')
    rgr = mean_red / mean_green
    rgr = np.nan_to_num(rgr, nan=0.0, posinf=8.0, neginf=0.0)

    return rgr


# Normalized Difference Nitrogen Index (NDNI): 7
def calculate_ndni(reflectance_cube, wavelengths):
    """
    Calculate the Normalized Difference Nitrogen Index (NDNI).

    NDNI = [log(R1510) - log(R1680)] / [log(R1510) + log(R1680)]

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D NDNI array.
    """
    wavelengths = np.array(wavelengths)

    # Find band indices closest to 1510 nm and 1680 nm
    idx_1510 = np.argmin(np.abs(wavelengths - 1510))
    idx_1680 = np.argmin(np.abs(wavelengths - 1680))

    # Extract reflectance bands
    band_1510 = reflectance_cube[idx_1510, :, :]
    band_1680 = reflectance_cube[idx_1680, :, :]

    # Avoid log of zero or negative values by setting a small floor
    epsilon = 1e-6
    band_1510 = np.maximum(band_1510, epsilon)
    band_1680 = np.maximum(band_1680, epsilon)

    # Compute log10 values
    log_1510 = np.log10(band_1510)
    log_1680 = np.log10(band_1680)

    # Calculate NDNI
    np.seterr(divide='ignore', invalid='ignore')
    numerator = log_1510 - log_1680
    denominator = log_1510 + log_1680
    ndni = numerator / denominator
    ndni = np.clip(ndni, 0.0, 1.0)

    return ndni

#----------------------------------------
# Dry or Senescent Carbon 7
#----------------------------------------

# Normalized Difference Lignin Index (NDLI): 7
def calculate_ndli(reflectance_cube, wavelengths):
    """
    Calculate the Normalized Difference Lignin Index (NDLI).

    NDLI = [log(R1754) - log(R1680)] / [log(R1754) + log(R1680)]

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D NDLI array.
    """
    wavelengths = np.array(wavelengths)

    # Find band indices closest to 1754 nm and 1680 nm
    idx_1754 = np.argmin(np.abs(wavelengths - 1754))
    idx_1680 = np.argmin(np.abs(wavelengths - 1680))

    # Extract reflectance bands
    band_1754 = reflectance_cube[idx_1754, :, :]
    band_1680 = reflectance_cube[idx_1680, :, :]

    # Avoid log of zero or negative values by setting a small floor
    epsilon = 1e-6
    band_1754 = np.maximum(band_1754, epsilon)
    band_1680 = np.maximum(band_1680, epsilon)

    # Compute log10 values
    log_1754 = np.log10(band_1754)
    log_1680 = np.log10(band_1680)

    # Calculate NDLI
    np.seterr(divide='ignore', invalid='ignore')
    numerator = log_1754 - log_1680
    denominator = log_1754 + log_1680
    ndli = numerator / denominator
    ndli = np.clip(ndli, 0.0, 1.0)

    return ndli

# Cellulose Absorption Index (CAI): 8
def calculate_cai(reflectance_cube, wavelengths):
    """
    Calculate the Cellulose Absorption Index (CAI).

    CAI = 0.5 * (R2000 + R2200) - R2100

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D CAI array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 2000nm, 2100nm, and 2200nm bands
    idx_2000 = np.argmin(np.abs(wavelengths - 2000))
    idx_2100 = np.argmin(np.abs(wavelengths - 2100))
    idx_2200 = np.argmin(np.abs(wavelengths - 2200))

    # Extract reflectance bands
    band_2000 = reflectance_cube[idx_2000, :, :]
    band_2100 = reflectance_cube[idx_2100, :, :]
    band_2200 = reflectance_cube[idx_2200, :, :]

    # Calculate CAI
    cai = 0.5 * (band_2000 + band_2200) - band_2100

    return cai


# Plant Senescence Reflectance Index (PSRI): 8
def calculate_psri(reflectance_cube, wavelengths):
    """
    Calculate the Plant Senescence Reflectance Index (PSRI).

    PSRI = (R680 - R500) / R750

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D PSRI array.
    """
    wavelengths = np.array(wavelengths)

    # Find band indices closest to 500nm, 680nm, and 750nm
    idx_500 = np.argmin(np.abs(wavelengths - 500))
    idx_680 = np.argmin(np.abs(wavelengths - 680))
    idx_750 = np.argmin(np.abs(wavelengths - 750))

    # Extract reflectance bands
    band_500 = reflectance_cube[idx_500, :, :]
    band_680 = reflectance_cube[idx_680, :, :]
    band_750 = reflectance_cube[idx_750, :, :]

    # Calculate PSRI
    np.seterr(divide='ignore', invalid='ignore')
    psri = (band_680 - band_500) / band_750
    psri = np.clip(psri, -1.0, 1.0)

    return psri



# Carotenoid Reflectance Index 1 (CRI1): 8
def calculate_cri1(reflectance_cube, wavelengths):
    """
    Calculate the Carotenoid Reflectance Index 1 (CRI1).

    CRI1 = (1 / R510) - (1 / R550)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D CRI1 array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 510nm and 550nm bands
    idx_510 = np.argmin(np.abs(wavelengths - 510))
    idx_550 = np.argmin(np.abs(wavelengths - 550))

    # Extract reflectance bands
    band_510 = reflectance_cube[idx_510, :, :]
    band_550 = reflectance_cube[idx_550, :, :]

    # Avoid division by zero or very low reflectance values
    epsilon = 1e-6
    band_510 = np.maximum(band_510, epsilon)
    band_550 = np.maximum(band_550, epsilon)

    # Calculate CRI1
    cri1 = (1.0 / band_510) - (1.0 / band_550)
    cri1 = np.clip(cri1, 0.0, 15.0)  # Clip to reasonable max value

    return cri1


# Carotenoid Reflectance Index 2 (CRI2): 8
def calculate_cri2(reflectance_cube, wavelengths):
    """
    Calculate the Carotenoid Reflectance Index 2 (CRI2).

    CRI2 = (1 / R510) - (1 / R700)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D CRI2 array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 510nm and 700nm bands
    idx_510 = np.argmin(np.abs(wavelengths - 510))
    idx_700 = np.argmin(np.abs(wavelengths - 700))

    # Extract reflectance bands
    band_510 = reflectance_cube[idx_510, :, :]
    band_700 = reflectance_cube[idx_700, :, :]

    # Avoid divide-by-zero
    epsilon = 1e-6
    band_510 = np.maximum(band_510, epsilon)
    band_700 = np.maximum(band_700, epsilon)

    # Calculate CRI2
    cri2 = (1.0 / band_510) - (1.0 / band_700)
    cri2 = np.clip(cri2, 0.0, 15.0)  # Clip to expected range

    return cri2


# Anthocyanin Reflectance Index 1 (ARI1): 9
def calculate_ari1(reflectance_cube, wavelengths):
    """
    Calculate the Anthocyanin Reflectance Index 1 (ARI1).

    ARI1 = (1 / R550) - (1 / R700)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D ARI1 array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 550nm and 700nm
    idx_550 = np.argmin(np.abs(wavelengths - 550))
    idx_700 = np.argmin(np.abs(wavelengths - 700))

    # Extract reflectance bands
    band_550 = reflectance_cube[idx_550, :, :]
    band_700 = reflectance_cube[idx_700, :, :]

    # Avoid divide-by-zero
    epsilon = 1e-6
    band_550 = np.maximum(band_550, epsilon)
    band_700 = np.maximum(band_700, epsilon)

    # Calculate ARI1
    ari1 = (1.0 / band_550) - (1.0 / band_700)
    ari1 = np.clip(ari1, 0.0, 0.2)

    return ari1

# Anthocyanin Reflectance Index 2 (ARI2): 9
def calculate_ari2(reflectance_cube, wavelengths):
    """
    Calculate the Anthocyanin Reflectance Index 2 (ARI2).

    ARI2 = R800 * [(1 / R550) - (1 / R700)]

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D ARI2 array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 550nm, 700nm, and 800nm
    idx_550 = np.argmin(np.abs(wavelengths - 550))
    idx_700 = np.argmin(np.abs(wavelengths - 700))
    idx_800 = np.argmin(np.abs(wavelengths - 800))

    # Extract reflectance bands
    band_550 = reflectance_cube[idx_550, :, :]
    band_700 = reflectance_cube[idx_700, :, :]
    band_800 = reflectance_cube[idx_800, :, :]

    # Avoid divide-by-zero
    epsilon = 1e-6
    band_550 = np.maximum(band_550, epsilon)
    band_700 = np.maximum(band_700, epsilon)

    # Calculate ARI2
    ari2 = band_800 * ((1.0 / band_550) - (1.0 / band_700))
    ari2 = np.clip(ari2, 0.0, 0.2)

    return ari2

#----------------------------------------
# Canopy Water Content 9
#----------------------------------------

# Water Band Index (WBI): 9
def calculate_wbi(reflectance_cube, wavelengths):
    """
    Calculate the Water Band Index (WBI).

    WBI = R900 / R970

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D WBI array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 900nm and 970nm bands
    idx_900 = np.argmin(np.abs(wavelengths - 900))
    idx_970 = np.argmin(np.abs(wavelengths - 970))

    # Extract reflectance bands
    band_900 = reflectance_cube[idx_900, :, :]
    band_970 = reflectance_cube[idx_970, :, :]

    # Avoid divide-by-zero
    epsilon = 1e-6
    band_970 = np.maximum(band_970, epsilon)

    # Calculate WBI
    wbi = band_900 / band_970
    wbi = np.clip(wbi, 0.0, 2.0)  # Broad clip; typical values: 0.8–1.2

    return wbi


# Normalized Difference Water Index (NDWI): 9
def calculate_ndwi(reflectance_cube, wavelengths):
    """
    Calculate the Normalized Difference Water Index (NDWI).

    NDWI = (R857 - R1241) / (R857 + R1241)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D NDWI array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 857nm and 1241nm bands
    idx_857 = np.argmin(np.abs(wavelengths - 857))
    idx_1241 = np.argmin(np.abs(wavelengths - 1241))

    # Extract reflectance bands
    band_857 = reflectance_cube[idx_857, :, :]
    band_1241 = reflectance_cube[idx_1241, :, :]

    # Calculate NDWI
    np.seterr(divide='ignore', invalid='ignore')
    ndwi = (band_857 - band_1241) / (band_857 + band_1241)
    ndwi = np.clip(ndwi, -1.0, 1.0)

    return ndwi

# Moisture Stress Index (MSI): 10
def calculate_msi(reflectance_cube, wavelengths):
    """
    Calculate the Moisture Stress Index (MSI).

    MSI = R1599 / R819

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D MSI array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 1599nm and 819nm bands
    idx_1599 = np.argmin(np.abs(wavelengths - 1599))
    idx_819 = np.argmin(np.abs(wavelengths - 819))

    # Extract reflectance bands
    band_1599 = reflectance_cube[idx_1599, :, :]
    band_819 = reflectance_cube[idx_819, :, :]

    # Avoid divide-by-zero
    epsilon = 1e-6
    band_819 = np.maximum(band_819, epsilon)

    # Calculate MSI
    msi = band_1599 / band_819
    msi = np.clip(msi, 0.0, 3.0)  # Clip to expected range

    return msi

# Normalized Difference Infrared Index (NDII): 10
def calculate_ndii(reflectance_cube, wavelengths):
    """
    Calculate the Normalized Difference Infrared Index (NDII).

    NDII = (R819 - R1649) / (R819 + R1649)

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D NDII array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 819nm and 1649nm bands
    idx_819 = np.argmin(np.abs(wavelengths - 819))
    idx_1649 = np.argmin(np.abs(wavelengths - 1649))

    # Extract reflectance bands
    band_819 = reflectance_cube[idx_819, :, :]
    band_1649 = reflectance_cube[idx_1649, :, :]

    # Calculate NDII
    np.seterr(divide='ignore', invalid='ignore')
    ndii = (band_819 - band_1649) / (band_819 + band_1649)
    ndii = np.clip(ndii, -1.0, 1.0)

    return ndii

# Triangular Vegetation Index (TVI) 10
def calculate_tvi(reflectance_cube, wavelengths):
    """
    Calculate the Triangular Vegetation Index (TVI).

    TVI = 0.5 * [120 * (R750 - R550) - 200 * (R670 - R550)]

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D TVI array.
    """
    wavelengths = np.array(wavelengths)

    # Get indices for 550nm (green), 670nm (red), 750nm (NIR)
    idx_550 = np.argmin(np.abs(wavelengths - 550))
    idx_670 = np.argmin(np.abs(wavelengths - 670))
    idx_750 = np.argmin(np.abs(wavelengths - 750))

    # Extract reflectance bands
    band_550 = reflectance_cube[idx_550, :, :]
    band_670 = reflectance_cube[idx_670, :, :]
    band_750 = reflectance_cube[idx_750, :, :]

    # Calculate TVI
    tvi = 0.5 * (120 * (band_750 - band_550) - 200 * (band_670 - band_550))

    return tvi


# Triangular Greenness Index (TGI) 11
def calculate_tgi(reflectance_cube, wavelengths):
    """
    Calculate the Triangular Greenness Index (TGI).

    TGI = -0.5 * [190 * (R670 - R550) - 120 * (R670 - R480)]

    Parameters:
    reflectance_cube (np.ndarray): 3D array (bands, rows, cols) of reflectance or RGB-like values.
    wavelengths (list or np.ndarray): 1D array of wavelength values corresponding to the bands.

    Returns:
    np.ndarray: 2D TGI array.
    """
    wavelengths = np.array(wavelengths)

    # Find indices for 480nm (blue), 550nm (green), and 670nm (red)
    idx_480 = np.argmin(np.abs(wavelengths - 480))
    idx_550 = np.argmin(np.abs(wavelengths - 550))
    idx_670 = np.argmin(np.abs(wavelengths - 670))

    # Extract reflectance bands
    band_480 = reflectance_cube[idx_480, :, :]
    band_550 = reflectance_cube[idx_550, :, :]
    band_670 = reflectance_cube[idx_670, :, :]

    # Calculate TGI
    tgi = -0.5 * (190 * (band_670 - band_550) - 120 * (band_670 - band_480))

    return tgi


#-----------------------------
# Chromaticity
#-----------------------------
def compute_chromaticity(hypercube, wavelengths, ranges):
    """
    Compute chromaticity components from hyperspectral data.

    Parameters:
    - hypercube: np.ndarray of shape (bands, rows, cols)
    - wavelengths: list or array of wavelengths (same length as bands)
    - ranges: list of 3 tuples, each defining a (min, max) wavelength range for chromaticity bands

    Returns:
    - chromaticity_cube: np.ndarray of shape (3, rows, cols)
    """
    chroma_bands = []
    for rmin, rmax in ranges:
        band_indices = np.where((wavelengths >= rmin) & (wavelengths <= rmax))[0]
        avg_band = np.mean(hypercube[band_indices, :, :], axis=0)
        chroma_bands.append(avg_band)

    # Stack into a (3, rows, cols) cube
    chroma_bands = np.stack(chroma_bands, axis=0)
    
    # Compute chromaticity
    intensity = np.mean(chroma_bands, axis=0)
    epsilon = 1e-6  # prevent divide by zero
    chromaticity = chroma_bands / (intensity + epsilon)

    return chromaticity

#------------------------------
# Band Ratio
#------------------------------
# import numpy as np
# from spectral import open_image
# import matplotlib.pyplot as plt

# 1. Load ENVI data
# import numpy as np
# from spectral import open_image

def load_envi_image(hdr_path):
    """
    Load a hyperspectral image in ENVI format (.hdr and associated binary file).
    
    Parameters:
        hdr_path (str): Path to the .hdr file.
    
    Returns:
        data (np.ndarray): 3D array of shape (rows, cols, bands) with reflectance data.
        wavelengths (np.ndarray): 1D array of wavelength values (nm) for each band.
        metadata (dict): Dictionary of image metadata from the .hdr file.
    """
    img = open_image(hdr_path)
    data = img.load()  # lazy-loaded reflectance cube
    wavelengths = np.array([float(w) for w in img.metadata['wavelength']])
    metadata = img.metadata
    return data, wavelengths, metadata


def get_band_index(target_wavelength, wavelengths):
    """
    Find the index of the band closest to a target wavelength.

    Parameters:
        target_wavelength (float): Desired wavelength (in nm).
        wavelengths (np.ndarray): 1D array of wavelength values.

    Returns:
        int: Index of the closest wavelength band.
    """
    return np.abs(wavelengths - target_wavelength).argmin()


def compute_band_ratio(data, wavelengths, num_wavelength, den_wavelength):
    """
    Compute a band ratio: data at num_wavelength / data at den_wavelength.

    Parameters:
        data (np.ndarray): Hyperspectral image array with shape (rows, cols, bands).
        wavelengths (np.ndarray): 1D array of wavelengths for each band.
        num_wavelength (float): Wavelength for the numerator band.
        den_wavelength (float): Wavelength for the denominator band.

    Returns:
        np.ndarray: 2D band ratio image.
    """
    band_num = get_band_index(num_wavelength, wavelengths)
    band_den = get_band_index(den_wavelength, wavelengths)
    num = data[:, :, band_num].astype(np.float32)
    den = data[:, :, band_den].astype(np.float32)
    ratio = np.where(den != 0, num / den, np.nan)
    return ratio




import os
import numpy as np
import rasterio
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from rasterio.crs import CRS


def parse_hdr_file(hdr_file_path):
    metadata = {}
    with open(hdr_file_path, 'r') as hdr_file:
        for line in hdr_file:
            if "=" in line:
                key, value = line.strip().split("=", 1)
                key = key.strip().lower()
                value = value.strip()
                if value.startswith("{") and value.endswith("}"):
                    value = value.strip("{}").split(",")
                    parsed_value = []
                    for v in value:
                        v = v.strip()
                        if v.replace(".", "", 1).isdigit() or v.replace("-", "", 1).replace(".", "", 1).isdigit():
                            parsed_value.append(float(v))
                        else:
                            parsed_value.append(v)
                    value = parsed_value
                elif value.isdigit():
                    value = int(value)
                elif value.replace(".", "", 1).isdigit():
                    value = float(value)
                metadata[key] = value
    return metadata


def extract_crs_from_map_info(map_info):
    try:
        crs_string = map_info[7].strip()
        if crs_string.upper() == "WGS84":
            return CRS.from_epsg(4326)
        else:
            return CRS.from_string(crs_string)
    except Exception:
        return None


def load_binary_data(binary_file_path, hdr_metadata):
    dtype_map = {
        4: np.float32,
        5: np.float64,
        1: np.uint8,
        2: np.int16,
        12: np.uint16,
        3: np.int32,
    }
    dtype = dtype_map[hdr_metadata['data type']]
    bands = hdr_metadata['bands']
    samples = hdr_metadata['samples']
    lines = hdr_metadata['lines']
    interleave = hdr_metadata.get("interleave", "bsq").lower()

    data = np.fromfile(binary_file_path, dtype=dtype)

    if interleave == "bsq":
        return data.reshape((bands, lines, samples))
    elif interleave == "bil":
        return data.reshape((lines, bands, samples)).transpose(1, 0, 2)
    elif interleave == "bip":
        return data.reshape((lines, samples, bands)).transpose(2, 0, 1)
    else:
        raise ValueError(f"Unsupported interleave format: {interleave}")


def load_pixel_mask(mask_path):
    with rasterio.open(mask_path) as src:
        return src.read(1)


def apply_mask(data, mask):
    return data[:, mask == 1]


def select_pca_components(data_2d_scaled, threshold=0.95):
    pca = PCA().fit(data_2d_scaled)
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    return np.argmax(cumsum >= threshold) + 1


def perform_pca(data, mask=None, n_components=3, auto_select=False, threshold=0.95, nodata_value=-9999):
    """
    Perform PCA on hyperspectral data, with support for masking or automatic NoData detection.

    Parameters:
    - data: np.ndarray (bands, lines, samples)
    - mask: Optional 2D np.ndarray with shape (lines, samples)
    - n_components: Number of PCA components
    - auto_select: If True, determine number of components to explain `threshold` variance
    - threshold: Cumulative variance to preserve (used only if auto_select=True)
    - nodata_value: Value to write into invalid pixels

    Returns:
    - pca_result: PCA-transformed data
    - pca: fitted PCA model
    - mask_used: The final pixel mask used
    """
    bands, lines, samples = data.shape

    # Reshape data to (pixels, bands)
    data_2d = data.reshape(bands, -1).T  # (pixels, bands)

    # Infer mask if not provided: exclude all-zero or any-NaN pixels
    if mask is None:
        print("Inferring mask from data...")
        invalid = np.all(data_2d == 0, axis=1) | np.any(np.isnan(data_2d), axis=1)
        mask_flat = ~invalid
        mask_used = mask_flat.reshape(lines, samples)
    else:
        mask_flat = mask.flatten().astype(bool)
        mask_used = mask

    # Filter valid pixels
    data_valid = data_2d[mask_flat]

    # Scale
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_valid)

    # PCA
    if auto_select:
        n_components = select_pca_components(data_scaled, threshold)

    pca = PCA(n_components=n_components)
    data_pca = pca.fit_transform(data_scaled)  # (valid_pixels, components)

    # Reconstruct full image with nodata
    pca_image = np.full((n_components, lines * samples), nodata_value, dtype=np.float32)
    pca_image[:, mask_flat] = data_pca.T
    pca_image = pca_image.reshape(n_components, lines, samples)

    return pca_image, pca, mask_used



def save_pca_result(pca_result, meta, output_path, mask=None, nodata_value=-9999):
    pca_image = pca_result

    meta.update({
        "count": pca_image.shape[0],
        "dtype": "float32",
        "nodata": nodata_value,
    })

    with rasterio.open(output_path, 'w', **meta) as dest:
        dest.write(pca_image)
    print(f"PCA result saved to: {output_path}")


def plot_explained_variance(pca, output_path_prefix):
    ratios = pca.explained_variance_ratio_
    cumsum = np.cumsum(ratios)

    # Elbow plot
    plt.figure()
    plt.plot(range(1, len(ratios) + 1), cumsum, marker='o')
    plt.xlabel("Number of Principal Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("Explained Variance by PCA Components")
    plt.grid(True)
    plt.savefig(f"{output_path_prefix}_explained_variance.png")
    plt.close()

    # Individual component variance
    plt.figure()
    plt.bar(range(1, len(ratios) + 1), ratios)
    plt.xlabel("Principal Component")
    plt.ylabel("Variance Explained")
    plt.title("Variance Explained per Component")
    plt.grid(True)
    plt.savefig(f"{output_path_prefix}_variance_per_component.png")
    plt.close()


def process_pca_binary_core(data, hdr_metadata, output_path, mask=None, n_components=3,
                            auto_select=False, threshold=0.95, good_bands=None):
    if good_bands is not None:
        good_bands = good_bands[:data.shape[0]]
        data = data[good_bands]

    pca_result, pca_model, mask_used = perform_pca(data, mask, n_components, auto_select, threshold)

    meta = {
        "driver": "GTiff",
        "dtype": "float32",
        "count": pca_result.shape[0],
        "height": hdr_metadata["lines"],
        "width": hdr_metadata["samples"],
        "nodata": -9999,
        "crs": extract_crs_from_map_info(hdr_metadata["map info"]),
        "transform": rasterio.transform.from_origin(
            float(hdr_metadata["map info"][3]),
            float(hdr_metadata["map info"][4]),
            float(hdr_metadata["map info"][5]),
            float(hdr_metadata["map info"][6])
        )
    }

    save_pca_result(pca_result, meta, output_path, mask_used)

    output_prefix = os.path.splitext(output_path)[0]
    plot_explained_variance(pca_model, output_prefix)

    print("Explained Variance Ratios:", pca_model.explained_variance_ratio_)
    print("Cumulative Variance Explained:", np.cumsum(pca_model.explained_variance_ratio_))

def fix_headwall_metadata(meta):
    try:
        for i in [3, 4, 5, 6]:
            meta["map info"][i] = float(meta["map info"][i])
    except Exception as e:
        print(f"Warning: Failed to fix Headwall map info: {e}")
    return meta

def process_headwall_binary(data_file_path, hdr_file_path, output_path, mask_path=None,
                            n_components=3, auto_select=False, threshold=0.95, good_bands=None):
    hdr_metadata = parse_hdr_file(hdr_file_path)
    hdr_metadata = fix_headwall_metadata(hdr_metadata)
    data = load_binary_data(data_file_path, hdr_metadata)
    mask = load_pixel_mask(mask_path) if mask_path else None
    return process_pca_binary_core(data, hdr_metadata, output_path, mask,
                                   n_components, auto_select, threshold, good_bands)



def perform_pca_from_tif(input_tif_path, output_pca_path, n_components=3, good_bands=None, nodata_value=-9999):
    """
    Run PCA on a multi-band GeoTIFF and save the top principal components.

    Parameters:
    - input_tif_path: str, path to input multi-band GeoTIFF
    - output_pca_path: str, where to save the PCA result
    - n_components: int, number of PCA components to retain
    - good_bands: list or np.array of bool or indices, optional mask to select only some bands
    - nodata_value: float, value to write into invalid regions
    """
    with rasterio.open(input_tif_path) as src:
        data = src.read(masked=True).astype(np.float32)  # Reads with nodata masked
        profile = src.profile

    # Convert masked values to NaN so they can be excluded in PCA
    if isinstance(data, np.ma.MaskedArray):
        data = data.filled(np.nan)

    if good_bands is not None:
        data = data[good_bands]

    bands, height, width = data.shape
    flat = data.reshape(bands, -1).T  # (pixels, bands)

    # Mask invalid pixels (all zeros or any NaN across bands)
    mask_valid = ~(np.all(flat == 0, axis=1) | np.any(np.isnan(flat), axis=1))
    flat_valid = flat[mask_valid]

    scaler = StandardScaler()
    flat_scaled = scaler.fit_transform(flat_valid)

    pca = PCA(n_components=n_components)
    flat_pca = pca.fit_transform(flat_scaled)

    # Rebuild full image with nodata
    pca_image = np.full((n_components, height * width), nodata_value, dtype=np.float32)
    pca_image[:, mask_valid] = flat_pca.T
    pca_image = pca_image.reshape((n_components, height, width))

    profile.update({
        "count": n_components,
        "dtype": "float32",
        "nodata": nodata_value
    })

    with rasterio.open(output_pca_path, "w", **profile) as dst:
        dst.write(pca_image)

    # Plot variance explained
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(cumsum)+1), cumsum, marker='o')
    plt.xlabel("Number of Components")
    plt.ylabel("Cumulative Explained Variance")
    plt.title("PCA Explained Variance")
    plt.grid(True)
    plt.savefig(output_pca_path.replace(".tif", "_explained_variance.png"))
    plt.close()

    print(f"PCA complete. Output saved to: {output_pca_path}")
    print("Explained variance:", pca.explained_variance_ratio_)
    print("Cumulative variance explained:", cumsum)

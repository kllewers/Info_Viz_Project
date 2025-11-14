import numpy as np
import rasterio

def save_array_to_tif(data, output_path, reference_dataset, band_names=None, nodata_value=-9999):
    """
    Save single-band or multi-band data to a GeoTIFF file, with optional NoData handling.

    Parameters:
    data (np.ndarray or dict): Either a single 2D array or a dict of {band_name: 2D array}.
    output_path (str): File path to save the output TIFF.
    reference_dataset (rasterio.DatasetReader): Used to extract spatial metadata.
    band_names (list, optional): List of band names for multi-band data.
    nodata_value (float, optional): Value to use for invalid pixels. Default is -9999.
    """
    meta = reference_dataset.meta.copy()

    if isinstance(data, np.ndarray):
        # Single-band
        meta.update({
            "count": 1,
            "dtype": "float32",
            "nodata": nodata_value
        })

        # Infer mask: zero or NaN values
        masked_data = np.where(
            (data == 0) | np.isnan(data),
            nodata_value,
            data
        ).astype(np.float32)

        with rasterio.open(output_path, 'w', **meta) as dst:
            dst.write(masked_data, 1)

        print(f"Saved single-band TIFF: {output_path}")

    elif isinstance(data, dict):
        # Multi-band
        arrays = list(data.values())
        band_count = len(arrays)

        meta.update({
            "count": band_count,
            "dtype": "float32",
            "nodata": nodata_value
        })

        with rasterio.open(output_path, 'w', **meta) as dst:
            for idx, array in enumerate(arrays, start=1):
                masked_array = np.where(
                    (array == 0) | np.isnan(array),
                    nodata_value,
                    array
                ).astype(np.float32)
                dst.write(masked_array, idx)

        if band_names:
            print(f"Saved multi-band TIFF with bands: {', '.join(band_names)}")
        else:
            print(f"Saved multi-band TIFF with {band_count} unnamed bands: {output_path}")

    else:
        raise ValueError("Unsupported data format: must be a 2D array or dict of 2D arrays.")

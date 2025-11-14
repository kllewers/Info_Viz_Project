import geopandas as gpd
import rasterio
from rasterio.features import geometry_mask


def shapefile_to_tif(shapefile_path, reference_tif_path, output_tif_path):
    """Convert a shapefile to a GeoTIFF mask with the same dimensions as the reference raster."""
    
    # Step 1: Read the shapefile
    gdf = gpd.read_file(shapefile_path)

    # Step 2: Open the reference GeoTIFF to get metadata and bounds
    with rasterio.open(reference_tif_path) as src:
        meta = src.meta.copy()
        transform = src.transform
        out_shape = (src.height, src.width)

    # Step 3: Generate a geometry mask from the shapefile
    geoms = [feature["geometry"] for feature in gdf.iterfeatures()]
    mask = geometry_mask(geoms, transform=transform, invert=True, out_shape=out_shape)

    # Step 4: Save the mask as a new GeoTIFF
    meta.update({"count": 1, "dtype": "uint8"})
    with rasterio.open(output_tif_path, 'w', **meta) as dst:
        dst.write(mask.astype(rasterio.uint8), 1)

    print(f"Shapefile converted to TIFF mask and saved at {output_tif_path}")


def combine_masks(mask_list, method='and'):
    """Combine a list of masks using a logical 'and' or 'or' operation."""
    if not mask_list:
        raise ValueError("mask_list must contain at least one mask.")
    
    # Start with the first mask in the list
    combined_mask = mask_list[0]
    
    # Iterate over the rest of the masks (starting from index 1)
    for mask in mask_list[1:]:
        if method == 'and':
            combined_mask &= mask  # Logical AND
        elif method == 'or':
            combined_mask |= mask  # Logical OR
        else:
            raise ValueError("Method must be either 'and' or 'or'.")
    
    return combined_mask


def load_pixel_mask(mask_path):
    """Load the pixel mask from a GeoTIFF file."""
    with rasterio.open(mask_path) as src:
        mask = src.read(1)
    return mask

def apply_mask(data, mask):
    """Apply the pixel mask to the data."""
    return data[:, mask == 1]

def validate_mask_dimensions(mask, data_shape):
    """Validate that the mask and data dimensions match."""
    if mask.shape != data_shape[1:]:
        raise ValueError("Mask dimensions do not match data dimensions.")


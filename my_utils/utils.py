import ee
import geemap
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from shapely.affinity import scale, translate
from datetime import datetime
from skimage import exposure
import logging  # Added import
import warnings

# Configure logging to suppress info-level logs
logging.basicConfig(level=logging.ERROR)
logging.getLogger('ee').setLevel(logging.ERROR)  # Suppress 'ee' logs
logging.getLogger('geemap').setLevel(logging.ERROR)  # Suppress 'geemap' logs

# Refined warning filter to use warnings.DeprecationWarning
warnings.filterwarnings('ignore')


class Utils:
    @staticmethod
    def scale_geometry(geometry, scale_factor):
        """
        Scale a geometry object around its centroid.

        Parameters:
            geometry (shapely.geometry.base.BaseGeometry): Geometry to scale.
            scale_factor (float): Factor by which to scale the geometry.

        Returns:
            shapely.geometry.base.BaseGeometry: Scaled geometry object.
        """
        # Calculate centroid
        centroid = geometry.centroid
        # Translate geometry to origin
        translated_geometry = translate(geometry, -centroid.x, -centroid.y)
        # Scale geometry
        scaled_geometry = scale(translated_geometry, xfact=scale_factor, yfact=scale_factor, origin=(0, 0))
        # Translate back to the original centroid
        return translate(scaled_geometry, centroid.x, centroid.y)

    @staticmethod
    def format_date(date_str):
        """Convert date string to 'YYYY-MM-DD' format."""
        formats = ['%d-%m-%Y', '%Y-%m-%d %H:%M:%S']
        for fmt in formats:
            try:
                return datetime.strptime(date_str, fmt).strftime('%Y-%m-%d')
            except ValueError:
                continue
        logging.error(f"Invalid date format for {date_str}. Expected one of {formats}.")
        return None

    @staticmethod
    def gdf_to_gee(gdf):
        """Converts a GeoDataFrame to an Earth Engine FeatureCollection."""
        features = []
        for _, row in gdf.iterrows():
            geom = row['geometry'].__geo_interface__  # Convert geometry to GeoJSON format
            feature = ee.Feature(ee.Geometry(geom), row.to_dict())  # Create an EE Feature
            features.append(feature)
        return ee.FeatureCollection(features)

    @staticmethod
    def download_rgb_image(collection_name: str, bands: list, start_date: str, end_date: str, region: ee.Geometry, output_folder: str = 'downloads') -> str:
        """Download RGB bands from a GEE collection filtered by date and region."""
        # Load the image collection, filter by date, and clip to region
        collection = ee.ImageCollection(collection_name).filterDate(start_date, end_date).filterBounds(region)
        image = collection.sort('system:time_start', False).first().select(bands).clip(region)  # Most recent image

        # Define unique filename based on image dates
        image_id = image.id().getInfo() or f'image_{start_date}_{end_date}'
        image_name = f'{output_folder}/{image_id}_RGB_{start_date}_{end_date}.tif'

        # Suppress stdout and stderr during image export
        geemap.ee_export_image(
                image,
                filename=image_name,
                scale=10,  # Sentinel-2 resolution in meters
                region=region,
                file_per_band=False,  # Save as a multi-band TIFF
                crs='EPSG:4326'
            )
        
        logging.warning(f"Downloaded: {image_name}")
        return image_name

    @staticmethod
    def process_row(index, row):
        try:
            # Format start and end dates
            start_date = Utils.format_date(row['SDate'])
            end_date = Utils.format_date(row['HDate'])

            # Skip rows with invalid dates
            if not start_date or not end_date:
                logging.warning(f"Skipping entry due to invalid dates: SDate={row['SDate']}, HDate={row['HDate']}")
                return index, None

            # Extract and check geometry
            region_geometry = row['geometry']
            district = row['District']

            # Verify valid geometry and convert it to EE format
            if region_geometry.is_empty:
                logging.warning(f"Skipping entry due to empty geometry for District: {district}")
                return index, None

            # Convert geometry type to EE compatible format
            if region_geometry.geom_type == 'Polygon':
                region = ee.Geometry.Polygon(region_geometry.__geo_interface__['coordinates'])
            elif region_geometry.geom_type == 'MultiPolygon':
                coords = [polygon.exterior.coords[:] for polygon in region_geometry.geoms]
                region = ee.Geometry.MultiPolygon(coords)
            else:
                logging.warning(f"Skipping unsupported geometry type: {region_geometry.geom_type} for District: {district}")
                return index, None

            # Define Sentinel-2 collection and bands
            image_collection_name = 'COPERNICUS/S2'
            bands = ['B4', 'B3', 'B2', 'B8', 'B5', 'B6', 'B7', 'B8A', 'B11', 'B12']

            # Attempt to download the image with retries

            image_file_path = Utils.download_rgb_image(
                image_collection_name,
                bands,
                start_date,
                end_date,
                region,
                output_folder='downloads'
            )
            return index, image_file_path
        except Exception as e:
            logging.error(f"Error processing index {index} for District {district}: {e}", exc_info=True)
            return index, None

    @staticmethod
    def scale_band(band):
        """
        Scales pixel values of a single band to the 0-255 range.

        Parameters:
        - band: np.array, pixel values of the band

        Returns:
        - np.array, scaled pixel values in the 0-255 range
        """
        band = band.astype(np.float32)  # Ensure values are in float for scaling
        return 255 * (band - np.min(band)) / (np.max(band) - np.min(band))  # Scale to 0–255

    @staticmethod
    def visualize_rgb_image(file_path, gamma=0.6, contrast_stretch=True):
        """
        Visualize an RGB image using matplotlib with scaling, optional gamma correction, and contrast stretching.

        Parameters:
        - file_path: str, path to the RGB image file (GeoTIFF)
        - gamma: float, gamma correction factor (default=0.6)
        - contrast_stretch: bool, whether to apply contrast stretching (default=True)
        """
        # Open the image file
        with rasterio.open(file_path) as src:
            # Read RGB bands (assuming Sentinel-2 band order: Red=B4, Green=B3, Blue=B2)
            red = src.read(3)  # Band 4 for Red
            green = src.read(2)  # Band 3 for Green
            blue = src.read(1)  # Band 2 for Blue

            # Scale each band to the 0–255 range for better visualization
            red_scaled = Utils.scale_band(red)
            green_scaled = Utils.scale_band(green)
            blue_scaled = Utils.scale_band(blue)

            # Stack the scaled RGB bands into a single image
            rgb = np.dstack((red_scaled, green_scaled, blue_scaled)).astype(np.uint8)

            # Apply contrast stretching if specified
            if contrast_stretch:
                p2, p98 = np.percentile(rgb, (2, 98))  # Calculate 2nd and 98th percentiles for stretching
                rgb = exposure.rescale_intensity(rgb, in_range=(p2, p98))

            # Apply gamma correction to adjust brightness
            rgb = exposure.adjust_gamma(rgb, gamma=gamma)

            # Display the processed image using matplotlib
            plt.figure(figsize=(5, 5))
            plt.imshow(rgb)
            plt.axis('off')  # Hide axes for a cleaner look
            plt.title("RGB Composite (Red-Green-Blue) with Scaling, Contrast Stretch, and Gamma Correction")
            plt.show()

        return rgb

    # Feature calculation functions
    @staticmethod
    def calculate_ndvi(nir_band, red_band):
        """Calculate NDVI (Normalized Difference Vegetation Index)."""
        ndvi = (nir_band - red_band) / (nir_band + red_band)
        return np.nanmean(ndvi)

    @staticmethod
    def calculate_evi(nir_band, red_band, blue_band):
        """Calculate EVI (Enhanced Vegetation Index)."""
        evi = 2.5 * (nir_band - red_band) / (nir_band + 6 * red_band - 7.5 * blue_band + 1)
        return np.nanmean(evi)

    @staticmethod
    def calculate_ndwi(nir_band, green_band):
        """Calculate NDWI (Normalized Difference Water Index)."""
        ndwi = (green_band - nir_band) / (green_band + nir_band)
        return np.nanmean(ndwi)

    @staticmethod
    def calculate_gndvi(nir_band, green_band):
        """Calculate GNDVI (Green Normalized Difference Vegetation Index)."""
        gndvi = (nir_band - green_band) / (nir_band + green_band)
        return np.nanmean(gndvi)

    @staticmethod
    def calculate_savi(nir_band, red_band, L=0.5):
        """Calculate SAVI (Soil Adjusted Vegetation Index)."""
        savi = ((nir_band - red_band) / (nir_band + red_band + L)) * (1 + L)
        return np.nanmean(savi)

    @staticmethod
    def calculate_msavi(nir_band, red_band):
        """Calculate MSAVI (Modified Soil Adjusted Vegetation Index)."""
        msavi = (2 * nir_band + 1 - np.sqrt((2 * nir_band + 1)**2 - 8 * (nir_band - red_band))) / 2
        return np.nanmean(msavi)

    @staticmethod
    def process_row_for_features(index, row):
        features = {'index': index}

        # Retrieve the TIFF file path and check if it's valid
        tif_path = row['tif_path']
        if not isinstance(tif_path, str):
            logging.warning(f"Skipping entry due to missing tif_path for index {index}")
            # Add NaN for all features if path is missing
            features.update({
                'ndvi': np.nan,
                'evi': np.nan,
                'ndwi': np.nan,
                'gndvi': np.nan,
                'savi': np.nan,
                'msavi': np.nan
            })
            return features

        # Open the TIFF file and read bands for feature calculation
        with rasterio.open(tif_path) as src:
            red = src.read(3)    # B4 for Red
            green = src.read(2)  # B3 for Green
            blue = src.read(1)   # B2 for Blue
            nir = src.read(4)    # B8 for NIR

            # Calculate each feature
            features['ndvi'] = Utils.calculate_ndvi(nir, red)
            features['evi'] = Utils.calculate_evi(nir, red, blue)
            features['ndwi'] = Utils.calculate_ndwi(nir, green)
            features['gndvi'] = Utils.calculate_gndvi(nir, green)
            features['savi'] = Utils.calculate_savi(nir, red)
            features['msavi'] = Utils.calculate_msavi(nir, red)

        return features
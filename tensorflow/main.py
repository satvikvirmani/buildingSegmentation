from osgeo import gdal

# Path to your TIFF file
file_path = "path/to/your/file.tif"

# Open the file
dataset = gdal.Open(file_path)

if dataset is None:
    print("Failed to open the file.")
else:
    print(f"Raster size: {dataset.RasterXSize} x {dataset.RasterYSize}")
    print(f"Number of bands: {dataset.RasterCount}")
    print(f"Projection: {dataset.GetProjection()}")

    # Get information about the first band
    band = dataset.GetRasterBand(1)
    print(f"Band Type: {gdal.GetDataTypeName(band.DataType)}")

    # Read raster data
    data = band.ReadAsArray()
    print(f"Data Shape: {data.shape}")

    # Clean up
    dataset = None
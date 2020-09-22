# Superpixel Segmentation

This provides an advanced walkthrough of how to obtain superpixel segments using multiple remote sensing imagery.

We do the following:

1. Download Hansen and ALOS-1 data, despeckling the latter so it can be used for meaningful segments
2. Reproject into an area determined by a shapefile determining an ROI
3. Segment the data using skimage carefully keeping track of nodata areas
4. Aggregate statistics within each superpixel segment and save this data to tif and shapefiles.

The reason this is advanced is because a) the datasets are slightly larger requiring more time to process and b) we are carefully keeping track of nodata areas. This is easy conceptually, but is not readily dealt with in the scikit ecosystem.
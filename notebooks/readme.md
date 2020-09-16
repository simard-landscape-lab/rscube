# Tutorials

The tutorials demonstrate how to leverage python to do basic remote sensing analysis, especially with SAR data.
By no means are these the "only" way of doing something and many of the methods shown here can be adapted and improved upon.

1. [Supervised Changed Detection](supervised_classification)
   + Collect rasters (as geotiffs) across a region of interest. For the demonstration, we use ALOS-1 backscatter imagery and Hansen Landsat mosaics.
   + Draw shape files over our region of interest to indicate land cover classes
   + Extract superpixel segments to obtain a segmentatin of our region of interest.
   + Classify superpixel segments according to the hand-drawn training data over region of interest.

2. [Change Detection](change_detection) (more advanced)
   + Collect SAR time series of imagery. Currently, our demo has:
     + Gulf coast changes with UAVSAR
     + Borreal forests in Quebec using ALOS-1
     + Mangrove Changes in Malaysia using ALOS-1
   + Use a statistic related to a hypothesis test expounded here to obtain a change criterion.
   + Apply change criterion across time series and save detected changes in common GIS formats.
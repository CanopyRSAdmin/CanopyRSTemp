import os
import argparse
from pathlib import Path

import rasterio
from geodataset.utils import get_utm_crs
from rasterio.crs import CRS
import geopandas as gpd
from shapely.geometry import box


def process_raster(tif_path: str, output_folder: str):
    # 1) Read raster bounds + CRS
    with rasterio.open(tif_path) as src:
        bounds = src.bounds  # left, bottom, right, top
        src_crs = src.crs

    # Build extent polygon in source CRS
    extent_poly = box(bounds.left, bounds.bottom, bounds.right, bounds.top)
    gdf_extent = gpd.GeoDataFrame({"geometry": [extent_poly]}, crs=src_crs)

    # 2) Reproject to UTM if needed so that units are meters
    working_gdf = gdf_extent
    if not src_crs.is_projected:
        # centroid in lon/lat
        centroid_lon = (bounds.left + bounds.right) / 2.0
        centroid_lat = (bounds.bottom + bounds.top) / 2.0
        utm_crs = get_utm_crs(centroid_lon, centroid_lat)
        working_gdf = gdf_extent.to_crs(utm_crs)

    # 3) Compute central vertical band in working CRS
    # (top-left corner, area = 15%, width ≤ 40 m)
    minx, miny, maxx, maxy = working_gdf.total_bounds
    full_w = maxx - minx
    full_h = maxy - miny
    total_area = full_w * full_h
    target_area = 0.15 * total_area

    # pick a width up to 40 m (or the raster width)
    w_cand = min(40.0, full_w)
    h_cand = target_area / w_cand

    if h_cand <= full_h:
        band_w, band_h = w_cand, h_cand
    else:
        # if that’d exceed the raster’s height, cap to full height
        band_h = full_h
        band_w = target_area / band_h

    # build the top-left box
    band_minx = minx
    band_maxx = minx + band_w
    band_maxy = maxy
    band_miny = maxy - band_h
    band_poly = box(band_minx, band_miny, band_maxx, band_maxy)

    # 4) Carve out the "train" AOI = extent minus band
    train_poly = working_gdf.iloc[0].geometry.difference(band_poly)

    # 5) If we reprojected, bring both back to the original CRS
    if not src_crs.is_projected:
        band_gdf = (
            gpd.GeoDataFrame({"geometry": [band_poly]}, crs=working_gdf.crs)
            .to_crs(src_crs)
        )
        train_gdf = (
            gpd.GeoDataFrame({"geometry": [train_poly]}, crs=working_gdf.crs)
            .to_crs(src_crs)
        )
    else:
        # no reprojection needed
        band_gdf = gpd.GeoDataFrame({"geometry": [band_poly]}, crs=src_crs)
        train_gdf = gpd.GeoDataFrame({"geometry": [train_poly]}, crs=src_crs)

    # 6) Write out two .gpkg files
    raster_name = os.path.splitext(os.path.basename(tif_path))[0]
    train_fp = f"{output_folder}/{raster_name}_aoi_train.gpkg"
    valid_fp = f"{output_folder}/{raster_name}_aoi_valid.gpkg"

    train_gdf.to_file(train_fp, driver="GPKG")
    band_gdf.to_file(valid_fp, driver="GPKG")

    print(f"Written training AOI (extent minus band) to: {train_fp}")
    print(f"Written validation AOI (central band) to: {valid_fp}")

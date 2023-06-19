import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import glob
from osgeo import gdal, osr, gdalconst
import subprocess
import rasterio
from rasterio.plot import show
import geemap
import ee
import geetools
import geopandas as gpd
import rioxarray as rxr
from rio_cogeo.profiles import cog_profiles
from ipyleaflet import Map, Marker, basemaps, basemap_to_tiles, DrawControl
import matplotlib as mpl
import seaborn as sns
import statsmodels
import json
from sklearn.linear_model import LinearRegression
from scipy import stats

def maskS2clouds(image: ee.Image) -> ee.Image:
    """
    Masks clouds, snow, and shadows in Sentinel-2 imagery using QA band information.

    Args:
        image: Sentinel-2 image to mask clouds, snow, and shadows.

    Returns:
        Masked Sentinel-2 image.
    """

    # Bitwise AND operations to create cloud, snow, and shadow masks
    cloud_mask = image.select('QA60').bitwiseAnd(int('11111', 10)).eq(0)
    snow_mask = image.select('QA60').bitwiseAnd(int('11111', 11)).eq(0)
    shadow_mask = image.select('QA60').bitwiseAnd(int('11111', 3)).eq(0)

    # Apply masks and add optical bands
    optical_bands = image.select('B.').multiply(0.0001)
    masked_image = (
        image.addBands(optical_bands, None, True)
        .updateMask(cloud_mask)
        .updateMask(snow_mask)
        .updateMask(shadow_mask)
    )

    return masked_image


def remove_sunglint(image: ee.Image, glint_geo: ee.Geometry) -> ee.Image:
    """
    Remove sunglint from an image using the glint removal technique.

    Args:
        image: The input image with sunglint.
        glint_geo: Geometry to clip the image for slope calculation.

    Returns:
        The image with sunglint removed.

    Raises:
        ee.EEException: If an error occurs during Earth Engine computation.
    """
    # Band selection
    B2 = image.select(['B8', 'B2'])
    B3 = image.select(['B8', 'B3'])
    B4 = image.select(['B8', 'B4'])
    B8 = image.select(['B8'])

    # Linear fit reduction
    lfitB2 = B2.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=10,
        bestEffort=True
    )
    lfitB3 = B3.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=10,
        bestEffort=True
    )
    lfitB4 = B4.reduceRegion(
        reducer=ee.Reducer.linearFit(),
        geometry=glint_geo,
        scale=10,
        bestEffort=True
    )

    # Extract slope values
    slope_B2 = ee.Image.constant(lfitB2.get('scale')).clip(glint_geo).rename('slope_B2')
    slope_B3 = ee.Image.constant(lfitB3.get('scale')).clip(glint_geo).rename('slope_B3')
    slope_B4 = ee.Image.constant(lfitB4.get('scale')).clip(glint_geo).rename('slope_B4')

    # Extract minimum B8 value
    min_B8 = ee.Image.constant(image.select('B8').reduceRegion(
        ee.Reducer.min(),
        geometry=glint_geo,
        scale=10
    ).get('B8')).rename('min_B8')

    # Create glint factors image
    glint_factors = ee.Image([slope_B2, slope_B3, slope_B4, min_B8])
    image_add = image.addBands(glint_factors)

    # Perform deglinting
    deglint_B2 = image_add.expression(
        'Blue - (Slope * (NIR - MinNIR))', {
        'Blue': image_add.select('B2'),
        'NIR': image_add.select('B8'),
        'MinNIR': image_add.select('min_B8'),
        'Slope': image_add.select('slope_B2')
    }).rename('B2')

    deglint_B3 = image_add.expression(
        'Green - (Slope * (NIR - MinNIR))', {
        'Green': image_add.select('B3'),
        'NIR': image_add.select('B8'),
        'MinNIR': image_add.select('min_B8'),
        'Slope': image_add.select('slope_B3')
    }).rename('B3')

    deglint_B4 = image_add.expression(
        'Red - (Slope * (NIR - MinNIR))', {
        'Red': image_add.select('B4'),
        'NIR': image_add.select('B8'),
        'MinNIR': image_add.select('min_B8'),
        'Slope': image_add.select('slope_B4')
    }).rename('B4')

    # Create deglinted image
    image_deglint = ee.Image([deglint_B2, deglint_B3, deglint_B4, B8])

    return image_deglint


def remove_sunglint_map(image_collection: ee.ImageCollection, glint_geo: ee.Geometry) -> ee.ImageCollection:
    """
    Remove sunglint from an image collection using the glint removal technique.

    Args:
        image_collection: The input image collection with sunglint.
        glint_geo: Geometry to clip the images for slope calculation.

    Returns:
        The image collection with sunglint removed.

    Raises:
        ee.EEException: If an error occurs during Earth Engine computation.
    """
    def remove_sunglint_from_image(image):
        # Band selection
        B1 = image.select(['B8', 'B1'])
        B2 = image.select(['B8', 'B2'])
        B3 = image.select(['B8', 'B3'])
        B4 = image.select(['B8', 'B4'])

        # Linear fit reduction
        lfitB2 = B2.reduceRegion(
            reducer=ee.Reducer.linearFit(),
            geometry=glint_geo,
            scale=10,
            bestEffort=True
        )
        lfitB3 = B3.reduceRegion(
            reducer=ee.Reducer.linearFit(),
            geometry=glint_geo,
            scale=10,
            bestEffort=True
        )
        lfitB4 = B4.reduceRegion(
            reducer=ee.Reducer.linearFit(),
            geometry=glint_geo,
            scale=10,
            bestEffort=True
        )

        # Extract slope values
        slope_B2 = ee.Image.constant(lfitB2.get('scale')).clip(glint_geo).rename('slope_B2')
        slope_B3 = ee.Image.constant(lfitB3.get('scale')).clip(glint_geo).rename('slope_B3')
        slope_B4 = ee.Image.constant(lfitB4.get('scale')).clip(glint_geo).rename('slope_B4')

        # Extract minimum B8 value
        min_B8 = ee.Image.constant(image.select('B8').reduceRegion(
            ee.Reducer.min(),
            geometry=glint_geo,
            scale=10,
            bestEffort=True
        ).get('B8')).rename('min_B8')

        # Create glint factors image
        glint_factors = ee.Image([slope_B2, slope_B3, slope_B4, min_B8])
        image_add = image.addBands(glint_factors)

        # Perform deglinting
        deglint_B2 = image_add.expression(
            'Blue - (Slope * (NIR - MinNIR))', {
            'Blue': image_add.select('B2'),
            'NIR': image_add.select('B8'),
            'MinNIR': image_add.select('min_B8'),
            'Slope': image_add.select('slope_B2')
        }).rename('B2')

        deglint_B3 = image_add.expression(
            'Green - (Slope * (NIR - MinNIR))', {
            'Green': image_add.select('B3'),
            'NIR': image_add.select('B8'),
            'MinNIR': image_add.select('min_B8'),
            'Slope': image_add.select('slope_B3')
        }).rename('B3')

        deglint_B4 = image_add.expression(
            'Red - (Slope * (NIR - MinNIR))', {
            'Red': image_add.select('B4'),
            'NIR': image_add.select('B8'),
            'MinNIR': image_add.select('min_B8'),
            'Slope': image_add.select('slope_B4')
        }).rename('B4')

        # Create deglinted image
        image_deglint = ee.Image([deglint_B2, deglint_B3, deglint_B4, image.select('B5')])

        return image_deglint

    # Map the function over the image collection
    deglinted_collection = image_collection.map(remove_sunglint_from_image)

    return deglinted_collection

def calculate_sdb(
    S2_comp: ee.Image,
    pol: ee.Geometry,
    bath: ee.Image
) -> ee.Image:
    """
    Calculate the Sea Depth Bathymetry (SDB) using linear regression.

    Args:
        S2_comp: Sentinel-2 composite image.
        pol: Geometry to clip the images for linear regression.
        S2S_sr_log: Sentinel-2 surface reflectance logarithm image.
        bath: Bathymetry image.

    Returns:
        The SDB image.

    Raises:
        ee.EEException: If an error occurs during Earth Engine computation.
    """
    # Take the logarithm and clip the Sentinel-2 composite image
    S2_log = S2_comp.log().clip(pol)

    # Add the bathymetry image to the Sentinel-2 surface reflectance logarithm image
    S2_model = S2_log.addBands(bath.clip(pol).rename('lidar'))

    constant = ee.Image(1)
    xVar = S2_model.select(['B2', 'B3'])
    yVar = S2_model.select('lidar')

    imgRegress = ee.Image.cat(constant, xVar, yVar)

    # Perform robust linear regression
    linearRegression = imgRegress.reduceRegion(
        reducer=ee.Reducer.robustLinearRegression(numX=3, numY=1),
        geometry=pol,
        scale=10,
        bestEffort = True
    )

    coefList = ee.Array(linearRegression.get('coefficients')).toList()
    b0 = ee.List(coefList.get(0)).get(0).getInfo()
    b1 = ee.List(coefList.get(1)).get(0).getInfo()
    b2 = ee.List(coefList.get(2)).get(0).getInfo()

    print('OLS estimates:', linearRegression.getInfo())
    print('y-intercept:', b0)
    print('Slope (B2):', b1)
    print('Slope (B3):', b2)

    # Create images for the coefficients
    b0_int = ee.Image(b0).clip(pol).rename('Intercept')
    b2_slope = ee.Image(b1).clip(pol).rename('B2 Slope')
    b3_slope = ee.Image(b2).clip(pol).rename('B3 Slope')

    # Select B2 and B3 bands from the logarithm image
    b2_sr = ee.Image(S2_log.select('B2')).clip(pol).rename('B2')
    b3_sr = ee.Image(S2_log.select('B3')).clip(pol).rename('B3')

    # Create image with bathymetry factors
    bath_factors = ee.Image.cat(b0_int, b2_slope, b3_slope, b2_sr, b3_sr)

    # Calculate SDB using the bathymetry factors
    SDB = bath_factors.expression(
        'Intercept + (B2_Slope * Blue) + (B3_Slope * Red)',
        {
            'Blue': bath_factors.select('B2'),
            'Red': bath_factors.select('B3'),
            'B2_Slope': bath_factors.select('B2 Slope'),
            'B3_Slope': bath_factors.select('B3 Slope'),
            'Intercept': bath_factors.select('Intercept')
        }
    ).rename('SDB')

    return SDB
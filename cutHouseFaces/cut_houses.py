import requests
from pyproj import Transformer
import cv2
import rasterio
import numpy as np
from pathlib import Path
import os

def __wgs84_to_utm32(lat, lon):
    # Define source and destination CRSs
    source_crs = 'epsg:4326'
    target_crs = 'epsg:32632'

    # Create a transform object
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    utm_x, utm_y = transformer.transform(lon, lat)
    
    return [utm_x, utm_y]

def __utm32_to_wgs84(utm_x, utm_y):
    # Define source and destination CRSs
    source_crs = 'epsg:32632'
    target_crs = 'epsg:4326'

    # Create a transform object
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    lon, lat = transformer.transform(utm_x, utm_y)
    
    return [lat, lon]

def __overpass_query(bbox_wgs):
    overpass_endpoint = "https://overpass-api.de/api/interpreter"

    query = f"""
                    [out:json][timeout:30];
                    (
                    way["building"]({bbox_wgs[0]}, {bbox_wgs[1]}, {bbox_wgs[2]}, {bbox_wgs[3]});
                    relation["building"]["type"="multipolygon"]({bbox_wgs[0]}, {bbox_wgs[1]}, {bbox_wgs[2]}, {bbox_wgs[3]});
                    )
                    ;
                    out geom;
                    >;
                    out geom; 
                """

    # Define the Overpass query
    query_params = {
        'data': query,
        'format': 'json'
    }

    # Make the Overpass API request
    response = requests.get(overpass_endpoint, params=query_params)

    if response.status_code == 200:
        
        return response.json()

    else:
        print(f"Error: Unable to fetch data from Overpass API. Status code: {response.status_code}")
        return None

def __json_to_polygons(json_data, bbox):

    # Define source and destination CRSs
    source_crs = 'epsg:4326'
    target_crs = 'epsg:32632'
    transformer = Transformer.from_crs(source_crs, target_crs, always_xy=True)
    
    outer = []
    inner = []
    elements = []
    
    for element in json_data['elements']:
        elements.append(element['type']) if element['type'] not in elements else elements
        if element['type'] == 'way':
            coordinates = []
            for geometry in element['geometry']:
                utm_x, utm_y = transformer.transform(geometry['lon'], geometry['lat'])
                coordinates.append([utm_x-bbox[0], utm_y-bbox[1]])
            outer.append(coordinates)

        elif element['type'] == 'relation':
            for member in element['members']:
                coordinates = []
                for geometry in member['geometry']:
                    utm_x, utm_y = transformer.transform(geometry['lon'], geometry['lat'])
                    coordinates.append([utm_x-bbox[0], utm_y-bbox[1]])
                if member['role'] == 'outer':
                    outer.append(coordinates)
                elif member['role'] == 'inner':
                    inner.append(coordinates)

    return [outer, inner]

def cut_houses_from_tif(image_path, output='and', show_images=True):

    try:
        # Read the image
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1000, 1000))
        im = rasterio.open(image_path)
        bbox = im.bounds
        
        bbox_top_left = __utm32_to_wgs84(bbox[0], bbox[1])
        bbox_bottom_right = __utm32_to_wgs84(bbox[2], bbox[3])

        jsonData = __overpass_query([bbox_top_left[0], bbox_top_left[1], bbox_bottom_right[0], bbox_bottom_right[1]])

        house_polygons = __json_to_polygons(jsonData, bbox)

        # Create a mask image with the same shape as the input image
        mask_outer = np.zeros_like(image)
        mask_inner = np.zeros_like(image)

        for p in house_polygons[0]:
            p = np.array(p, dtype=np.int32)
            cv2.fillPoly(mask_outer, [p], (255, 255, 255))

        for p in house_polygons[1]:
            p = np.array(p, dtype=np.int32)
            cv2.fillPoly(mask_inner, [p], (255, 255, 255))

        mask_outer = cv2.flip(mask_outer, 0)
        mask_inner = cv2.flip(mask_inner, 0)

        mask_combined = cv2.bitwise_and(mask_outer, (cv2.bitwise_not(mask_inner)))

        # Bitwise NOT operation to invert the mask
        inverse_mask_outer = cv2.bitwise_not(mask_outer)
        inverse_mask_inner = cv2.bitwise_not(mask_inner)
        inverse_mask_combined = cv2.bitwise_not(mask_combined)

        if output == 'or':
            cutout_image_outer = cv2.bitwise_or(image, inverse_mask_outer)
            cutout_image_inner = cv2.bitwise_and(image, inverse_mask_inner)
            cutout_image_combined = cv2.bitwise_or(image, inverse_mask_combined)
        else:
            cutout_image_outer = cv2.bitwise_and(image, inverse_mask_outer)
            cutout_image_inner = cv2.bitwise_or(image, inverse_mask_inner)
            cutout_image_combined = cv2.bitwise_and(image, inverse_mask_combined)

        if(show_images):
            cv2.imshow('Original Image', image)
            # cv2.imshow('Cutout Image Outer', cutout_image_outer)
            # cv2.imshow('Cutout Image Inner', cutout_image_inner)
            cv2.imshow('Cutout Image Combined', cutout_image_combined)
            cv2.waitKey(0)

        path, file = os.path.split(image_path)
        output_path = f'{path}/{Path(image_path).stem}_masked.tif'
        cv2.imwrite(output_path, cutout_image_combined)
        
    except Exception as e:
        print(e)



if __name__ == "__main__":
    
    cut_houses_from_tif('Luftbilder/32686_5337.tif', show_images=False)

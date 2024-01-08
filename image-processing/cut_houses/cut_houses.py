import requests
from pyproj import Transformer
import cv2
import rasterio
import numpy as np
from pathlib import Path
import os
from core.tiff_handler import _load_image, _save_image

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

def __json_to_polygons(json_data, bbox, scaleFactor:float = 1.0):

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
                coordinates.append([(utm_x-bbox[0])*scaleFactor, (utm_y-bbox[1])*scaleFactor])
            outer.append(coordinates)

        elif element['type'] == 'relation':
            for member in element['members']:
                coordinates = []
                for geometry in member['geometry']:
                    utm_x, utm_y = transformer.transform(geometry['lon'], geometry['lat'])
                    coordinates.append([(utm_x-bbox[0])*scaleFactor, (utm_y-bbox[1])*scaleFactor])
                if member['role'] == 'outer':
                    outer.append(coordinates)
                elif member['role'] == 'inner':
                    inner.append(coordinates)

    return [outer, inner]

def __cut_houses_from_tif(image_path, output='and', show_images=True):

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

### cut houses interface new:
        
#def create_house_masks(image: str, output: Optional[str] = None) -> ndarray if output is not none > write image to output
def create_house_masks(image:str, output:str = None,)->np.ndarray:

    if(Path(image).suffix == '.tif'): #check if image is .tif 

        try:
            # # Read the image
            cv2_image = cv2.imread(image)
            # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            # cv2_image = cv2.resize(cv2_image, (1000, 1000))
            rasterio_image = rasterio.open(image)
            bbox = rasterio_image.bounds

            scaleFactor = cv2_image.shape[0]/1000
            
            bbox_top_left = __utm32_to_wgs84(bbox[0], bbox[1])
            bbox_bottom_right = __utm32_to_wgs84(bbox[2], bbox[3])

            jsonData = __overpass_query([bbox_top_left[0], bbox_top_left[1], bbox_bottom_right[0], bbox_bottom_right[1]])

            house_polygons = __json_to_polygons(jsonData, bbox, scaleFactor=scaleFactor)

            # Create a mask image with the same shape as the input image
            mask_outer = np.zeros_like(cv2_image)
            mask_inner = np.zeros_like(cv2_image)

            # Outer Polygons
            for p in house_polygons[0]:
                p = np.array(p, dtype=np.int32)
                cv2.fillPoly(mask_outer, [p], (255, 255, 255))

            # Inner Polygons
            for p in house_polygons[1]:
                p = np.array(p, dtype=np.int32)
                cv2.fillPoly(mask_inner, [p], (255, 255, 255))

            mask_outer = cv2.flip(mask_outer, 0)
            mask_inner = cv2.flip(mask_inner, 0)

            # Combine Masks of inner and outer House-Polygons
            mask_combined = cv2.bitwise_and(mask_outer, (cv2.bitwise_not(mask_inner)))

            # Bitwise NOT operation to invert the mask
            inverse_mask_combined = cv2.bitwise_not(mask_combined)

            # Test show image
            # cutout_image_combined = cv2.bitwise_and(cv2_image, inverse_mask_combined)
            # cv2.imshow('Cutout Image Combined', cutout_image_combined)
            # cv2.imshow('Mask', inverse_mask_combined)
            # cv2.waitKey(0)
            # cv2.imwrite('Luftbilder/scaled.tif', cutout_image_combined)
            
            if(output != None):
                try:
                    cv2.imwrite(output, inverse_mask_combined)
                except Exception as ee:
                    print(f'Exception while writing to output path: {ee}')

            return inverse_mask_combined

        except Exception as e:
            print(f'Exception while creating house mask: {e}')

#def cut_mask_from_image(image: str | ndarray, mask: str | ndarray, output: Optional[str] = None, inverted: bool = False) -> ndarray
def cut_mask_from_image(image:str|np.ndarray, mask:str|np.ndarray, output:str = None, inverted:bool = False)-> np.ndarray:

    try:
        image, tif_meta = _load_image(image)
        
        if(type(mask) == str):
            mask = cv2.imread(mask)

        if(inverted):
            mask = invert_mask(mask=mask)
        #     cutout_image = cv2.bitwise_or(image, mask)
        # else:
        #     cutout_image = cv2.bitwise_and(image, mask)

        cutout_image = cv2.bitwise_and(image, mask)

        # Test show image
        # cv2.imshow('Cutout Image Combined', cutout_image)
        # cv2.waitKey(0)

        if output is not None:
            try:
                _save_image(cutout_image, output, tif_meta)
            except Exception as ee:
                print(f'Exception while writing to output path: {ee}')

        return cutout_image

    except Exception as e:
        print(e)

    return None
    
#def invert_mask(mask: str | ndarray, output: Optional[str] = None) -> ndarray
def invert_mask(mask:str|np.ndarray, output: str = None)-> np.ndarray:
    try:
        if(type(mask) == str):
            mask = cv2.imread(mask)

        inverse_mask = cv2.bitwise_not(mask)
        
        # Test show mask
        # im = cv2.imread('Luftbilder/32686_5337.tif')
        # cv2.imshow('Cutout Image Combined', cv2.bitwise_and(im, inverse_mask))
        # cv2.imshow('inverse mask', inverse_mask)
        # cv2.waitKey(0)

        if(output != None):
                try:
                    cv2.imwrite(output, inverse_mask)
                except Exception as ee:
                    print(f'Exception while writing to output path: {ee}')

        return inverse_mask
    
    except Exception as e:
        print(e)

#def cut_houses(image: str, output: Optional[str] = None, inverted: bool = False) -> ndarray
def cut_houses(image: str, output:str = None, inverted: bool = False)-> np.ndarray:
    try:
        
        mask = create_house_masks(image=image)

        cutout_image = cut_mask_from_image(image=image, mask=mask, inverted=inverted, output=output)

        # Test show image
        # cv2.imshow('Cutout Image', cutout_image)
        # cv2.waitKey(0)

        return cutout_image

    except Exception as e:
        print(e)
    
    return None

def filename_append(path: str, insert: str) -> str:
    """
    Function inserts the 'insert' before the suffix of the 'path'
    Example: masked_path(path: 'Luftbilder/32686_5337.tif', insert: '_masked')
                --> 'Luftbilder/32686_5337_masked.tif'
    """
    path, file = os.path.split(image_path)
    output_path = f'{path}/{Path(image_path).stem}{insert}{Path(image_path).suffix}'
    return output_path

if __name__ == "__main__":
    
    image_path = 'Luftbilder/32686_5337.tif'

    # cut_houses_from_tif('Luftbilder/32686_5337.tif', show_images=False)
    # mask = create_house_masks(image='Luftbilder/32686_5337.tif', output='Luftbilder/32686_5337_mask.tif')
    # cutimage = cut_mask_from_image(image='Luftbilder/32686_5337.tif', mask=mask)
    # invertedmask = invert_mask(mask=mask)

    # cut_houses(image=image_path, output=masked_path(image_path, insert='masked_inv'), inverted=True)
    cut_houses(image=image_path, output=filename_append(image_path, insert='_masked_inv'), inverted=True)

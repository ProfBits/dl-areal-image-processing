import cv2
import numpy as np
import geojson
from pyproj import Transformer
import rasterio
import os
import overpass_import

def __load_geojsonPath(file_path):
    with open(file_path, encoding='utf8') as file:
        data = geojson.load(file)

    polygons = []
    for feature in data['features']:
        if feature['geometry']['type'] == 'Polygon':
            polygon = feature['geometry']['coordinates']
            polygons.append(polygon)

        elif feature['geometry']['type'] == 'MultiPolygon':
            multi_polygon = feature['geometry']['coordinates']
            polygons.extend(multi_polygon)

    return polygons

#didn't work
def __load_geojson_withLinestrings(file_path):
    with open(file_path, encoding='utf8') as file:
        data = geojson.load(file)

        polygons = []
        for feature in data['features']:
            geometry_type = feature['geometry']['type']
            coordinates = feature['geometry']['coordinates']

            if geometry_type == 'Polygon':
                polygons.append(coordinates)

            elif geometry_type == 'MultiPolygon':
                polygons.extend(coordinates)

            elif geometry_type == 'LineString' and coordinates[0] == coordinates[-1]:
                # Add LineString to polygons if the first and last coordinates are the same
                polygons.append(coordinates)
    return polygons

def geojsonPath_to_utmPolys(file_path, x, y):

    polygons = __load_geojsonPath(file_path)

    utm_polygons = []

    transformer = Transformer.from_crs('epsg:4326', 'epsg:32632', always_xy=True)

    for polygon in polygons:
        utm_polygon = []
        for tup in polygon:
            if len(tup[0]) == 2:
                lon, lat = zip(*tup)
                utm_x, utm_y = transformer.transform(lon, lat)
                utm_ring = list(zip(utm_x, utm_y))
                utm_ring = np.asarray(utm_ring)
                utm_ring2 = []
                for r in utm_ring:
                    utm_ring2.append([(r[0]-x), (r[1]-y)])
                utm_polygon.append(utm_ring2)
            else:
                print("Skipping ring with more than two values:", tup)

        utm_polygons.append(utm_polygon)
        
    return utm_polygons

def geojsonFile_to_utmPolys(geojson_polys, x, y):

    # polygons = __load_geojsonPath(file_path)
    polygons = geojson_polys

    utm_polygons = []

    transformer = Transformer.from_crs('epsg:4326', 'epsg:32632', always_xy=True)

    for polygon in polygons:
        utm_polygon = []
        for tup in polygon:
            if len(tup[0]) == 2:
                lon, lat = zip(*tup)
                utm_x, utm_y = transformer.transform(lon, lat)
                utm_ring = list(zip(utm_x, utm_y))
                utm_ring = np.asarray(utm_ring)
                utm_ring2 = []
                for r in utm_ring:
                    utm_ring2.append([(r[0]-x), (r[1]-y)])
                utm_polygon.append(utm_ring2)
            else:
                print("Skipping ring with more than two values:", tup)

        utm_polygons.append(utm_polygon)

    return utm_polygons

def cut_geojson_tif(image_path, geojson_path, output_path, output='and'):

    try:
        # Read the image
        image = cv2.imread(image_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (1000, 1000))
        im = rasterio.open(image_path)
        bbox = im.bounds

        utm_polys = geojsonPath_to_utmPolys(geojson_path, bbox[0], bbox[1])

        # Create a mask image with the same shape as the input image
        mask = np.zeros_like(image)

        for utmpoly in utm_polys:
            for p in utmpoly:
                p = np.array(p, dtype=np.int32)
                cv2.fillPoly(mask, [p], (255, 255, 255))

        mask = cv2.flip(mask, 0)

        # Bitwise NOT operation to invert the mask
        inverse_mask = cv2.bitwise_not(mask)

        if output == 'or':
            cutout_image = cv2.bitwise_or(image, inverse_mask)
        else:
            cutout_image = cv2.bitwise_and(image, inverse_mask)

        cv2.imshow('Original Image', image)
        cv2.imshow('Cutout Image', cutout_image)
        cv2.waitKey(0)
        cv2.imwrite(output_path, cutout_image)
    except Exception as e:
        print(e)



if __name__ == "__main__":
    
    image_path = 'Luftbilder/32691_5335.tif'
    # geojson_path = 'Hausumringe/691_5335.geojson'
    geojson_path = 'Hausumringe/691_5335_hh.geojson'
    """ split = os.path.splitext(image_path)
    vals = (split[0].split('/'))[-1]
    v = vals.split('_')
    x = v[0][2:]
    y = v[1] """
    output_path = 'Luftbilder/32691_5335_cut_hh.tif'

    cut_geojson_tif(image_path, geojson_path, output_path)
    
import overpy
import geojson
from decimal import Decimal

def __decimal_to_float(value):
    if isinstance(value, Decimal):
        return float(value)
    return value

def overpass_query_and_save(query, output_file):
    api = overpy.Overpass()

    result = api.query(query)

    # Convert Overpass result to GeoJSON
    features = []
    for way in result.ways:
        coordinates = [(__decimal_to_float(node.lon), __decimal_to_float(node.lat)) for node in way.nodes]
        
        # Create a GeoJSON polygon or multipolygon based on the geometry type
        if coordinates[0] == coordinates[-1]: 
            geometry = geojson.Polygon([coordinates])
        else:
            geometry = geojson.LineString(coordinates)

        feature = geojson.Feature(geometry=geometry, properties={"id": way.id})
        features.append(feature)

    feature_collection = geojson.FeatureCollection(features)

    # Save the GeoJSON data to a file
    with open(output_file, 'w') as file:
        geojson.dump(feature_collection, file)

def overpass_query_from_utm32(utm32, output_file):
    print()

if __name__ == "__main__":
    
    queryTxt = '[out:json][timeout:30];(way["building"](48.139558, 11.567501, 48.148245, 11.58138);relation["building"]["type"="multipolygon"](48.139558, 11.567501, 48.148245, 11.58138););out;>;out qt;'
    overpass_query_and_save(queryTxt, 'Hausumringe/691_5335_hh.geojson')

# Ideas

Suggesting by Auftraggebern

1. image
2. segments = segmentAnyting(image)
3. segment_feature = clip.encode_image(segment)
4. features = dim_reduce(segment_feature) <- only required for clustering as classification
5. class = classification(features) <- no labels

## Possible solutions

- Regular Imageprocessing (detect green pixels + noise reduction + shadow removal)
- Segment Anything + Clip Image Embeddings + Dimension Reduction + Clustering
- Segment Anything + Clip Image Embeddings + DNN (no labels issue)
- U-Net, Areal image directly to mask

## Processing Ideas

- Remove roofs (cut out)
- Remove road (cut out), issues with street trees
- Color filter/intensity
- Building/Road/Green/CityBlock masks
- Image cutting/gluing

## Work areas

- Building/Road/Green/CityBlock masks from existing maps
- U-Net
- Regular Imageprocessing
- Segment Anything
# Final results

The results where taken from `samgeo_batches_v7`

```
box_threshold: 0.25
text_threshold: 0.5
text_prompt: tree . lawn . gras
model_type: vit_h
tile_size: 256
overlap: 64
preprocessing: none
Houses were cut from predicted mask

Results:
   accuracy: 91.944 %
         f1:  0.901
     recall:  0.930 (or sensitivity)
  precision:  0.875
specificity:  0.913
        iou:  0.820 (or Jaccard index)

TPR: 0.930 | FNR: 0.070
TNR: 0.913 | FPR: 0.087
PPV: 0.875 | FDR: 0.125
NPV: 0.952 | FOR: 0.048

Prediction time: 0.56 hours
```

## Diff Overlay

False Positive shown in red and False Negative shown in blue

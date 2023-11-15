from segment_anything import SamPredictor, sam_model_registry

# sam = sam_model_registry["<model_type>"](checkpoint="<path/to/checkpoint>")
# predictor = SamPredictor(sam)
# predictor.set_image(<your_image>)
# masks, _, _ = predictor.predict(<input_prompts>)

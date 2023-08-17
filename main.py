from utils import *
import torch
import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from segment_anything import SamPredictor, sam_model_registry
from diffusers import StableDiffusionInpaintPipeline
from groundingdino.util.inference import load_model, load_image, predict, annotate
from GroundingDINO.groundingdino.util import box_ops


device = "cuda"

# Paths
sam_checkpoint_path = "./GroundingDINO/weights/sam_vit_h_4b8939.pth"
groundingdino_model_path = "./GroundingDINO/groundingdino/config/GroundingDINO_SwinT_OGC.py"
groundingdino_weights_path = "./GroundingDINO/weights/groundingdino_swint_ogc.pth"

# SAM Parameters
model_type = "vit_h"
sam_model = sam_model_registry[model_type](checkpoint=sam_checkpoint_path).to(device=device)
sam_predictor = SamPredictor(sam_model)

# Stable Diffusion
pipeline = StableDiffusionInpaintPipeline.from_pretrained("stabilityai/stable-diffusion-2-inpainting",
                                                     torch_dtype=torch.float16).to(device)

# Grounding DINO
groundingdino_model = load_model(groundingdino_model_path, groundingdino_weights_path)

def edit_image(path, item, prompt, box_threshold, text_threshold):
    """
    Edit an image by replacing objects using segmentation and inpainting.

    Args:
        path (str): Path to the image file.
        item (str): Object to be recognized in the image.
        prompt (str): Object to replace the selected object in the image.
        box_threshold (float): Threshold for bounding box predictions.
        text_threshold (float): Threshold for text predictions.

    Returns:
        np.ndarray: Edited image.
    """
    src, img = load_image(path)
    
    # Predict object bounding boxes, logits, and phrases
    boxes, logits, phrases = predict(
        model=groundingdino_model,
        image=img,
        caption=item,
        box_threshold=box_threshold,
        text_threshold=text_threshold
    )
    
    # Set up predictor
    sam_predictor.set_image(src)
    new_boxes = transform_boxes(sam_predictor,boxes, src,device)
    
    # Predict masks and annotations
    masks, _, _ = sam_predictor.predict_torch(
        point_coords=None,
        point_labels=None,
        boxes=new_boxes,
        multimask_output=False,
    )
    
    # Overlay mask on annotated image
    img_annotated_mask = show_mask(
        masks[0][0].cpu(),
        annotate(image_source=src, boxes=boxes, logits=logits, phrases=phrases)[...,::-1]
    )
    
    # Apply inpainting pipeline
    edited_image = pipeline(prompt=prompt,
                        image=Image.fromarray(src).resize((512, 512)),
                        mask_image=Image.fromarray(masks[0][0].cpu().numpy()).resize((512, 512))
    ).images[0]
    
    return edited_image

def main():

    img_path = "./car_img.jpg"

    Selected_Object = "fire hydrant"
    BOX_THRESHOLD = 0.3
    TEXT_THRESHOLD = 0.25

    Prompt = "Photo Booth"

    edited_image = edit_image(img_path,Selected_Object,Prompt,BOX_THRESHOLD,TEXT_THRESHOLD)

    save_image(edited_image,"edited_car.jpg")

if __name__ == "__main__":
    main()
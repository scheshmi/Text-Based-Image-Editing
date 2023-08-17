import numpy as np

from PIL import Image
from GroundingDINO.groundingdino.util import box_ops
import torch

def show_mask(mask, image, random_color=True):
    """
    Overlay a mask on an image and return the composited result.

    Args:
        mask (torch.Tensor): Mask to overlay.
        image (np.ndarray): Image to overlay the mask on.
        random_color (bool, optional): If True, overlay with random color. 
                                      If False, use a fixed color. Default is True.

    Returns:
        np.ndarray: Image with the mask overlaid.
    """
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.8])], axis=0)
    else:
        color = np.array([30/255, 144/255, 255/255, 0.6])

    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)

    annotated_frame_pil = Image.fromarray(image).convert("RGBA")
    mask_image_pil = Image.fromarray((mask_image.cpu().numpy() * 255).astype(np.uint8)).convert("RGBA")

    return np.array(Image.alpha_composite(annotated_frame_pil, mask_image_pil))

def transform_boxes(predictor,boxes, src,device):
    """
    Transform boxes to adjust to the source image dimensions.

    Args:
        boxes (torch.Tensor): Bounding boxes in the format [x_center, y_center, width, height].
        src (np.ndarray): Source image.

    Returns:
        torch.Tensor: Transformed boxes.
    """
    H, W, _ = src.shape
    boxes_xyxy = box_ops.box_cxcywh_to_xyxy(boxes) * torch.Tensor([W, H, W, H])
    return predictor.transform.apply_boxes_torch(boxes_xyxy, src.shape[:2]).to(device)

def save_image(image, file_path):
    """
    Save an image to the specified file path.
    
    Args:
        image (PIL.Image.Image): Image to be saved.
        file_path (str): Path where the image will be saved.
    """
    try:
        image.save(file_path)
        print(f"Image saved: {file_path}")
    except Exception as e:
        print(f"Error saving image to {file_path}: {e}")
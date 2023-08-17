# Text-Based-Image-Editing

Leveraging three computer vision foundation models, Segment Anything Model (SAM), Stable Diffusion, and Grounding DINO, to edit and manipulate images. Starting by leveraging Grounding DINO for zero-shot object detection driven by textual input. Then, using SAM, masks are extracted from the identified bounding boxes. These masks guide Stable Diffusion to replace the masked areas with contextually appropriate content derived from the text prompt, resulting in a cohesive text-based image editing process.

## Install Requirements
First, install requirements and Grounding DINO

```python 
pip install -r requirements.txt
```

Grounding DINO
```python 
git clone https://github.com/IDEA-Research/GroundingDINO.git
cd GroundingDINO/
pip install -e .
```

Run `download_files.py` to download pre-trained models
```python 
python download_files.py
```




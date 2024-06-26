# ToponymExtractor
 Automized toponym extraction pipeline for historical maps. This is the code repository for project "**Automated Toponym Extraction Pipeline for Historical Maps**" under Digital Humanities Lab, EPFL.

## Example Showcase
![toponym_extractor](/Imgs/toponym_extractor.jpg "illustration")

## Evaluation

 - ### Speed Benchmark
   - Config: Default Config
   - GPU: NVIDIA GeForce RTX 3070 Laptop GPU (8.0 GB VRAM, single GPU)
   - CPU: AMD Ryzen 7 5800H with Radeon Graphics (16 GB RAM)

| Resolution      | Toponyms | Word Spotting (sec) | Flattening (sec) | Toponym Assignment (sec) | Total (sec) | Total (mins) |
|-----------------|----------|---------------------|------------------|--------------------------|-------------|--------------|
| ~7.4k * 6.2k    | ~0.3k    | 33                  | 45               | 32                       | 120         | ~2           |
| ~8.2k * 6.3k    | ~3k      | 104                 | 218              | 45                     | 367         | ~6           |

	


## Usage
- ### Installation

```
git clone https://github.com/SesamePaste233/ToponymExtractor.git ToponymExtractor
cd ToponymExtractor
conda create -n toponymics python=3.8 -y
conda activate toponymics
pip install torch==1.9.0+cu111 torchvision==0.10.0+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
python -m pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cu111/torch1.9/index.html
cd WordSpotter
python setup.py build develop
```
After setting up the environment, the pipeline requires ***trained weights*** for neural networks in order to work properly. Put the weight files in ***./Models*** and ***configurate model paths*** when calling the pipeline!

- ### Trained Weights

All in one .7z package. Put the files extracted inside ***./Models*** .

**Google Drive**: [All In One](https://drive.google.com/file/d/153le_wEwPnzm8G566AWmDaC5aSmjabQR/view?usp=sharing) (1.23 G)

- ### Run on Maps
 <details>
 <summary>Simple Usage</summary>
 
 ```python
 from pipeline import ToponymExtractor
 
 cfg = {
     'img_path': 'path/to/map_file',

     # Model paths (default paths)
     'deepsolo_config_path': 'Models/config_96voc.yaml',
     'deepsolo_model_path': 'Models/finetune_v2/model.pth',
     
     'grouper_model_path': 'Models/grouper_model_v1_epoch2.pth',
 }
 
 extractor = ToponymExtractor(cfg)
 
 toponyms = extractor.run()
 ```
 
 </details>

 <details>
  
 <summary>All Configs</summary>
 
 ```python
 default_config = {
     # INPUT
     'img_path': None, # Overwrite this

     'task_name': None, # Overwrite this if needed

     'output_dir': 'Results/', # Overwrite this if needed

     # SETTINGS (default values work well for most cases)
     'pyramid_scan_num_layers': 1, # Significantly slows down detection speed
     'pyramid_min_patch_resolution': 384, # Lower this value for maps with smaller text
     'pyramid_max_patch_resolution': 2048, # Model only run on min_patch_resolution if pyramid_scan_num_layers = 1

     'word_spotting_score_threshold': 0.4,
     'word_spotting_image_batch_size': 8, # For 8G VRAM. Lower this value if CUDA OOM error occurs, increase it if you have a powerful GPU

     # Save intermediate results
     'save_stacked_detection': True,
     'save_flattened_detection': True,
     'save_grouper_graph': True,
     'save_toponym_detection': True,

     'save_visualization_images': True,

     # Model paths
     'deepsolo_config_path': 'Models/config_96voc.yaml',
     'deepsolo_model_path': 'Models/finetune_v2/model.pth',
     
     'grouper_model_path': 'Models/grouper_model_v1_epoch2.pth',

     # Optional
     'generate_style_embeddings': False,
     'use_style_embeddings_in_grouping': False,
     'deepfont_encoder_path': 'Models/DeepFontEncoder_full.pth',
 }
 ```
 
 </details>
 
## Credits
 - DeepSolo for word spotting:
   - Paper: [DeepSolo: Let Transformer Decoder with Explicit Points Solo for Text Spotting](https://arxiv.org/abs/2211.10772)
   - Repo: https://github.com/ViTAE-Transformer/DeepSolo
 - DeepFont for style recognition:
   - Paper: [DeepFont: Identify Your Font from An Image](https://arxiv.org/abs/1507.03196)



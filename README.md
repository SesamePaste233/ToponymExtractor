# ToponymExtractor
 Automized toponym extraction pipeline for historical maps. 

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
python setup.py build develop
```

- ### Run on Maps
 <details>
 <summary>Simple Usage</summary>
 
 ```python
 from pipeline import ToponymExtractor
 
 cfg = {
     'img_path': 'path/to/map.jpg',
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
   

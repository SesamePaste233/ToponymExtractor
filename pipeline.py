from WordSpotter.ModelWrapper import DeepSoloWrapper
from Grouper.GrouperCaller_v1 import *
from StyleEncoder.DeepFont import DeepFontEncoder, EncodeFontBatch, load_model

from WordSpotting import pyramid_scan
from Flattening import aggregate_closest_results, normalize_adhesive
from StyleEmbedding import generate_style_embeddings
from ToponymsAssignment import group_toponyms, toponym_from_graph_strong_component

import os
from PIL import Image
import time

from Utils import result_reader as rr
from Utils.visualizer import PolygonVisualizer

def get_default_config():
    return {
        # INPUT
        'img_path': None, # Overwrite this

        'task_name': None, # Overwrite this if needed

        'output_dir': 'Results/', # Overwrite this if needed

        # SETTINGS (default values work well for most cases)
        'pyramid_scan_num_layers': 1, # Significantly slows down detection speed
        'pyramid_min_patch_resolution': 384, # Lower this value for maps with smaller text
        'pyramid_max_patch_resolution': 2048, # Model only run on min_patch_resolution if pyramid_scan_num_layers = 1

        'word_spotting_score_threshold': 0.6, # 0 to 1, lower this value if some words are missed
        'word_spotting_image_batch_size': 8, # For 8G VRAM. Lower this value if CUDA OOM error occurs, increase it if you have a powerful GPU

        # Save intermediate results
        'save_stacked_detection': True,
        'save_flattened_detection': True,
        'save_grouper_graph': True,
        'save_toponym_detection': True,

        'save_visualization_images': False,

        # Model paths
        'deepsolo_config_path': 'Models/config_96voc.yaml',
        'deepsolo_model_path': 'Models/finetune_v2/model.pth',
        
        'grouper_model_path': 'Models/grouper_model_v1_epoch2.pth',

        # Optional
        'generate_style_embeddings': False,
        'use_style_embeddings_in_grouping': False,
        'deepfont_encoder_path': 'Models/DeepFontEncoder_full.pth',
    }

def merge_cfgs(default_cfg, user_cfg):
    for key, value in user_cfg.items():
        if key in default_cfg:
            default_cfg[key] = value
    
    if default_cfg['use_style_embeddings_in_grouping'] and not default_cfg['generate_style_embeddings']:
        default_cfg['use_style_embeddings_in_grouping'] = False
        print('use_style_embeddings_in_grouping is set to False because generate_style_embeddings is False')

    return default_cfg

class ToponymExtractor:
    def __init__(self, config:dict):
        if 'img_path' not in config or config['img_path'] is None:
            raise ValueError('img_path not found in config')

        default_config = get_default_config()
        config = merge_cfgs(default_config, config)

        self.config = config
        self.img_path = config['img_path']
        self.task_name = config.get('task_name', default_config['task_name'])
        self.config['task_name'] = self.task_name
        self.model_cfg = config.get('deepsolo_config_path', default_config['deepsolo_config_path'])
        self.model_weights = config.get('deepsolo_model_path', default_config['deepsolo_model_path'])
        self.grouper_model_path = config.get('grouper_model_path', default_config['grouper_model_path'])
        
        self.generate_style_embeddings = config.get('generate_style_embeddings', default_config['generate_style_embeddings'])
        self.use_style_embeddings_in_grouper = config.get('use_style_embeddings_in_grouping', default_config['use_style_embeddings_in_grouping'])
        self.deepfont_encoder_path = config.get('deepfont_encoder_path', default_config['deepfont_encoder_path'])

        self.pyramid_scan_num_layers = config.get('pyramid_scan_num_layers', default_config['pyramid_scan_num_layers'])
        self.pyramid_min_patch_resolution = config.get('pyramid_min_patch_resolution', default_config['pyramid_min_patch_resolution'])
        self.pyramid_max_patch_resolution = config.get('pyramid_max_patch_resolution', default_config['pyramid_max_patch_resolution'])

        self.word_spotting_score_threshold = config.get('word_spotting_score_threshold', default_config['word_spotting_score_threshold'])
        self.word_spotting_image_batch_size = config.get('word_spotting_image_batch_size', default_config['word_spotting_image_batch_size'])

        self._validate_paths()

        if self.task_name is None:
            self.task_name = os.path.splitext(os.path.basename(self.img_path))[0]

        # Outputs
        self.output_dir = config.get('output_dir', default_config['output_dir'])
        self.output_dir = os.path.join(self.output_dir, self.task_name)
        os.makedirs(self.output_dir, exist_ok=True)

        self.stacked_detection_path = os.path.join(self.output_dir, f'stacked_detections.json')
        self.save_stack_detection = config.get('save_stacked_detection', default_config['save_stacked_detection'])

        self.flattened_detection_path = os.path.join(self.output_dir, f'flattened_detections.json')
        self.save_flattened_detection = config.get('save_flattened_detection', default_config['save_flattened_detection'])

        self.grouper_graph_path = os.path.join(self.output_dir, f'grouper_graph.gexf')
        self.save_grouper_graph = config.get('save_grouper_graph', default_config['save_grouper_graph'])

        self.toponym_detection_path = os.path.join(self.output_dir, f'toponym_detections.json')
        self.save_toponym_detection = config.get('save_toponym_detection', default_config['save_toponym_detection'])

        self.save_visualization_images = config.get('save_visualization_images', default_config['save_visualization_images'])

        self._print_cfg()

    def _print_cfg(self):
        print('Configuration:')
        for key, value in self.config.items():
            print(f'{key}: \t\t\t{value}')

    def _validate_paths(self):
        if not os.path.exists(self.img_path):
            raise ValueError(f'img_path {self.img_path} does not exist')

        if not os.path.exists(self.model_cfg):
            raise ValueError(f'deepsolo_config_path {self.model_cfg} does not exist')

        if not os.path.exists(self.model_weights):
            raise ValueError(f'deepsolo_model_path {self.model_weights} does not exist')

        if not os.path.exists(self.grouper_model_path):
            raise ValueError(f'grouper_model_path {self.grouper_model_path} does not exist')

        if self.generate_style_embeddings and not os.path.exists(self.deepfont_encoder_path):
            raise ValueError(f'deepfont_encoder_path {self.deepfont_encoder_path} does not exist')

    def word_spotting(self):
        self.spotter = DeepSoloWrapper(self.model_cfg, self.model_weights, score_threshold=self.word_spotting_score_threshold)

        self.word_spotting_results = pyramid_scan(
            self.img_path, 
            self.stacked_detection_path, 
            self.spotter, 
            num_layers = self.pyramid_scan_num_layers, 
            min_patch_resolution=self.pyramid_min_patch_resolution,
            max_patch_resolution=self.pyramid_max_patch_resolution,
            spotting_batch_size=self.word_spotting_image_batch_size,
            save_visualization=self.save_visualization_images, 
            save_results=self.save_stack_detection
        )

        print(f'Word spotting done. Found {len(self.word_spotting_results)} raw word detections.')

    def flattening(self):
        self.grouped_results, self.ambiguity = aggregate_closest_results(self.word_spotting_results, sample_count=15, evaluate_overlapping="any")

        self.flattening_results = normalize_adhesive(self.grouped_results, self.ambiguity, Image.open(self.img_path))

        if self.save_flattened_detection:
            rr.save_json(self.flattening_results, self.flattened_detection_path)
        if self.save_visualization_images:
            vis = PolygonVisualizer()
            vis.canvas_from_image(Image.open(self.img_path))
            vis.draw(self.flattening_results).save(self.flattened_detection_path.replace('.json', '.jpg'))

        print(f'Flattening done. Found {len(self.flattening_results)} words.')

    def style_representing(self):
        deepfont_encoder = load_model(self.deepfont_encoder_path)

        self.flattening_results = generate_style_embeddings(self.flattening_results, Image.open(self.img_path), deepfont_encoder)
        
        if self.save_flattened_detection:
            rr.save_json(self.flattening_results, self.flattened_detection_path)

        print(f'Style embeddings generated.')

    def toponym_assignment(self):
        self.grouper = GrouperCaller(self.grouper_model_path)

        self.directed_graph, self.order_observations = group_toponyms(self.flattening_results, self.grouper, use_style_embeddings=self.use_style_embeddings_in_grouper, batch_size=128)

        if self.save_grouper_graph:
            rr.save_toponym_graph(self.directed_graph, self.grouper_graph_path)
            rr.save_json_nested(self.order_observations, self.grouper_graph_path.replace('.gexf', '.json'))

        self.toponyms_raw = toponym_from_graph_strong_component(self.flattening_results, self.directed_graph, self.order_observations)
        self.toponyms = rr.extract_toponyms_from_result_groups(self.toponyms_raw)

        if self.save_toponym_detection:
            rr.save_json_nested(self.toponyms, self.toponym_detection_path)
        if self.save_visualization_images:
            vis = PolygonVisualizer()
            vis.canvas_from_image(Image.open(self.img_path))
            vis.draw_toponyms(self.toponyms).save(self.toponym_detection_path.replace('.json', '.jpg'))

        print(f'Toponym assignment done. Found {len(self.toponyms)} toponyms.')

    def run(self, print_time=True):
        print(f'\nRunning pipeline for task: \'{self.task_name}\'.')

        start_time = time.time()

        self.word_spotting()
        step1_time = time.time()
        if print_time:
            print(f'Word spotting time: {step1_time - start_time:.2f} seconds')

        self.flattening()
        step2_time = time.time()
        if print_time:
            print(f'Flattening time: {step2_time - step1_time:.2f} seconds')

        if self.generate_style_embeddings:
            self.style_representing()
            step3_time = time.time()
            if print_time:
                print(f'Style representing time: {step3_time - step2_time:.2f} seconds')
        else:
            step3_time = time.time()

        self.toponym_assignment()
        end_time = time.time()
        if print_time:
            print(f'Toponym assignment time: {end_time - step3_time:.2f} seconds')

        print(f'Total time: {end_time - start_time:.2f} seconds')

        return self.toponyms
    
if __name__ == '__main__':
    if True:
        cfg = {
            'img_path': 'Test1/12148_btv1b530633990f1.jpg',
            # Save intermediate results
            'save_stacked_detection': True,
            'save_flattened_detection': True,
            'save_grouper_graph': True,
            'save_toponym_detection': True,

            'save_visualization_images': True,
        }

        extractor = ToponymExtractor(cfg)

        toponyms = extractor.run()

    if False:
        # Scan all images in Test1 folder
        import glob
        img_paths = glob.glob('Input/*')

        img_paths_contains = ['saunders_1874.jpeg', 'vandevelde_1846.jpeg']

        img_paths = [img_path for img_path in img_paths if any([img_path_contains in img_path for img_path_contains in img_paths_contains])]

        for img_path in img_paths:
            cfg = {
                'img_path': img_path,
                # Save intermediate results
                'save_stacked_detection': False,
                'save_flattened_detection': True,
                'save_grouper_graph': True,
                'save_toponym_detection': True,

                'save_visualization_images': True,
            }

            extractor = ToponymExtractor(cfg)

            toponyms = extractor.run()

    if False:
        import glob
        img_paths = glob.glob('rumsey/val/*')

        from Utils import result_reader as rr

        all_results = []

        for img_path in img_paths:
            cfg = {
                'img_path': img_path,
                # Save intermediate results
                'save_stacked_detection': False,
                'save_flattened_detection': True,
                'save_grouper_graph': True,
                'save_toponym_detection': True,

                'save_visualization_images': True,
            }

            extractor = ToponymExtractor(cfg)

            toponyms = extractor.run()

            rumsey_result = rr.to_rumsey_format(toponyms, img_path)

            all_results.append(rumsey_result)

        rr.save_json_nested(all_results, 'RumseyEval/rumsey_results.json')

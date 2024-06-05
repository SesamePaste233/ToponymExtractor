import cv2
import numpy as np

import torch
from detectron2.data import MetadataCatalog
from detectron2.engine.defaults import DefaultPredictor
from detectron2.utils.visualizer import ColorMode, Visualizer
from adet.utils.visualizer import TextVisualizer


from detectron2.modeling import build_model
from detectron2.checkpoint import DetectionCheckpointer
import detectron2.data.transforms as T
from adet.data.augmentation import Pad
from detectron2.data.detection_utils import convert_PIL_to_numpy
from adet.config import get_cfg

from adet.modeling import swin
from adet.modeling import vitae_v2

from PIL import Image

import sys

sys.path.append('../')

import Utils.bezier_utils as butils

class DeepSoloWrapper(object):
    def __init__(self, cfg_path, model_weights = None,instance_mode=ColorMode.IMAGE, score_threshold=0):
        """
        Args:
            cfg (CfgNode):
            model_weights (str): path to the model checkpoint, if None, use the model defined in cfg.
            instance_mode (ColorMode):
        """
        cfg = get_cfg()
        cfg.merge_from_file(cfg_path)
        self.metadata = MetadataCatalog.get(
            cfg.DATASETS.TEST[0] if len(cfg.DATASETS.TEST) else "__unused"
        )
        if model_weights is not None:
            cfg.MODEL.WEIGHTS = model_weights 
        cfg.freeze()

        self.cfg = cfg
        self.cpu_device = torch.device("cpu")
        self.instance_mode = instance_mode
        self.vis_text = cfg.MODEL.TRANSFORMER.ENABLED
        
        if cfg.MODEL.BACKBONE.NAME == "build_vitaev2_backbone":
            self.predictor = ViTAEBatchPredictor(cfg)
        else:
            self.predictor = BatchPredictor(cfg)

        self.images = []
        self.offset_xs = []
        self.offset_ys = []
        self.instances = []
        self.score_threshold = score_threshold

        self.voc_size = cfg.MODEL.TRANSFORMER.VOC_SIZE
        if self.voc_size == 96:
            self.CTLABELS = [' ','!','"','#','$','%','&','\'','(',')','*','+',',','-','.','/','0','1','2','3','4','5','6','7','8','9',':',';','<','=','>','?','@','A','B','C','D','E','F','G','H','I','J','K','L','M','N','O','P','Q','R','S','T','U','V','W','X','Y','Z','[','\\',']','^','_','`','a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','{','|','}','~']
        elif self.voc_size == 37:
            self.CTLABELS = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z','0','1','2','3','4','5','6','7','8','9']

    def run_on_image(self, image_bgr):
        return self.predictor(image_bgr)
        
    def run_on_PIL_image(self, image_pil):
        image_bgr = np.array(image_pil)[:, :, ::-1]
        return self.run_on_image(image_bgr)
    
    def load_batch(self, images, offset_xs = None, offset_ys = None):
        self.images.clear()
        self.offset_xs.clear()
        self.offset_ys.clear()
        self.instances.clear()
        self.combined_image = None
        
        self.images = images

        if offset_xs is None:
            offset_xs = [0] * len(images)
        if offset_ys is None:
            offset_ys = [0] * len(images)

        self.offset_xs = offset_xs
        self.offset_ys = offset_ys
        
        return

    def _ctc_decode_recognition(self, rec):
        last_char = '###'
        s = ''
        for c in rec:
            c = int(c)
            if c < self.voc_size - 1:
                if last_char != c:
                    if self.voc_size == 37 or self.voc_size == 96:
                        s += self.CTLABELS[c]
                        last_char = c
                    else:
                        s += str(chr(self.CTLABELS[c]))
                        last_char = c
            else:
                last_char = '###'
        return s

    def _interpret_instances(self, instances):
        instances = instances.to(self.cpu_device)
        ctrl_pnts = instances.ctrl_points.numpy()
        scores = instances.scores.tolist()
        recs = instances.recs
        bd_pnts = np.asarray(instances.bd)

        results = []

        for ctrl_pnt, score, rec, bd in zip(ctrl_pnts, scores, recs, bd_pnts):
            if score < self.score_threshold:
                continue
            polygon = []
            if bd is not None:
                bd = np.hsplit(bd, 2)
                polygon = np.vstack([bd[0], bd[1][::-1]])
            
            upper_line = polygon[:25]
            center_line = ctrl_pnt.reshape(-1, 2)
            lower_line = polygon[25:]

            upper_bezier_pts_x, upper_bezier_pts_y = butils.bezier_from_polyline_v2(upper_line[:, 0], upper_line[:, 1])
            center_bezier_pts_x, center_bezier_pts_y = butils.bezier_from_polyline_v2(center_line[:, 0], center_line[:, 1])
            lower_bezier_pts_x, lower_bezier_pts_y = butils.bezier_from_polyline_v2(lower_line[:, 0], lower_line[:, 1])

            text = self._ctc_decode_recognition(rec)

            results.append({
                "polygon_x": polygon[:, 0].tolist(),
                "polygon_y": polygon[:, 1].tolist(),
                "upper_bezier_pts": list(zip(upper_bezier_pts_x, upper_bezier_pts_y)),
                "center_bezier_pts": list(zip(center_bezier_pts_x, center_bezier_pts_y)),
                "lower_bezier_pts": list(zip(lower_bezier_pts_x, lower_bezier_pts_y)),
                "text": text,
                "score": score
            })

        return results

    def _offset_result_matrix(self, result, offset_matrix):
        xy = np.vstack([result["polygon_x"], result["polygon_y"], np.ones(len(result["polygon_x"]))])
        new_xy = np.dot(offset_matrix, xy)
        result["polygon_x"] = new_xy[0].tolist()
        result["polygon_y"] = new_xy[1].tolist()

        upper_bezier_pts = np.array(result["upper_bezier_pts"])
        center_bezier_pts = np.array(result["center_bezier_pts"])
        lower_bezier_pts = np.array(result["lower_bezier_pts"])

        upper_bezier_pts = np.dot(offset_matrix, np.vstack([upper_bezier_pts.T, np.ones(len(upper_bezier_pts))]))
        center_bezier_pts = np.dot(offset_matrix, np.vstack([center_bezier_pts.T, np.ones(len(center_bezier_pts))]))
        lower_bezier_pts = np.dot(offset_matrix, np.vstack([lower_bezier_pts.T, np.ones(len(lower_bezier_pts))]))

        result["upper_bezier_pts"] = upper_bezier_pts.T[:, :2]
        result["center_bezier_pts"] = center_bezier_pts.T[:, :2]
        result["lower_bezier_pts"] = lower_bezier_pts.T[:, :2]

        result["center"] = (np.mean(result["polygon_x"]), np.mean(result["polygon_y"]))
        result["left"] = center_bezier_pts.T[0, :2]
        result["right"] = center_bezier_pts.T[-1, :2]

        result["avg_height"] = (np.linalg.norm(np.array(result["upper_bezier_pts"][0]) - np.array(result["lower_bezier_pts"][-1])) + np.linalg.norm(np.array(result["upper_bezier_pts"][-1]) - np.array(result["lower_bezier_pts"][0])))/2

        return result

    def _offset_result(self, result, offset_x, offset_y, rot_center = None, rot_angle_degree = 0.0):
        
        def _offset_matrix(_offset_x, _offset_y, _rot_center, _rot_angle):
            offset_matrix = np.array([[1, 0, _offset_x], [0, 1, _offset_y], [0, 0, 1]])
            if _rot_center is not None:
                a = _rot_center[0]
                b = _rot_center[1]
                c = np.cos(_rot_angle)
                s = np.sin(_rot_angle)
                rot_matrix = np.array([[c, -s, a - a * c + b * s], [s, c, b - a * s - b * c], [0, 0, 1]])
                offset_matrix = np.dot(offset_matrix, rot_matrix)
            return offset_matrix

        offset_matrix = _offset_matrix(offset_x, offset_y, rot_center, np.deg2rad(rot_angle_degree))

        return self._offset_result_matrix(result, offset_matrix)

    def inference_single(self, image:Image.Image, offset_x = 0, offset_y = 0):
        size = image.size
        center = (size[0] / 2, size[1] / 2)
        image_np = convert_PIL_to_numpy(image, format="BGR")

        if isinstance(self.predictor, BatchPredictor):
            predictions = self.predictor([image_np])[0]
        else:
            predictions = self.predictor(image_np)
        
        instances = predictions["instances"]
        
        results = self._interpret_instances(instances)

        for result in results:
            result = self._offset_result(result, offset_x, offset_y)

        return results
    
    def inference_batch(self, images = None, offset_xs = None, offset_ys = None, batch_size = 1, rotations = [0], group_results = False):
        from tqdm import tqdm
        assert isinstance(self.predictor, BatchPredictor) or isinstance(self.predictor, ViTAEBatchPredictor), "This method is only available for BatchPredictor"

        if images is not None:
            self.load_batch(images, offset_xs, offset_ys)
        
        for rot in rotations:
            print(f"Rotating images by {rot} degrees")
            rot_center = (self.images[0].size[0] / 2, self.images[0].size[1] / 2)
            if rot == 0:
                image_nps = [convert_PIL_to_numpy(image, format="BGR") for image in self.images]
            else:
                image_nps = [convert_PIL_to_numpy(image.rotate(rot), format="BGR") for image in self.images]
            
            batches = [image_nps[i:i + batch_size] for i in range(0, len(image_nps), batch_size)]
            predictions = []
            for batch in tqdm(batches):
                predictions.extend(self.predictor(batch))

            for prediction, offset_x, offset_y in zip(predictions, self.offset_xs, self.offset_ys):
                instances = prediction["instances"]
                results = self._interpret_instances(instances)
                
                group = [self._offset_result(result, offset_x, offset_y, rot_center=rot_center, rot_angle_degree=rot) for result in results]

                if group_results:
                    self.instances.append(group)
                else:
                    self.instances.extend(group)

        return self.instances
        

class ViTAEPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        # each size must be divided by 32 with no remainder for ViTAE
        self.pad = Pad(divisible_size=32)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_image):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            if self.input_format == "RGB":
                original_image = original_image[:, :, ::-1]
            height, width = original_image.shape[:2]
            image = self.aug.get_transform(original_image).apply_image(original_image)
            image = self.pad.get_transform(image).apply_image(image)
            image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
            inputs = {"image": image, "height": height, "width": width}
            predictions = self.model([inputs])[0]
            return predictions
        
class ViTAEBatchPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )
        # each size must be divided by 32 with no remainder for ViTAE
        self.pad = Pad(divisible_size=32)

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_images):
        """
        Args:
            original_image (np.ndarray): an image of shape (H, W, C) (in BGR order).

        Returns:
            predictions (dict):
                the output of the model for one image only.
                See :doc:`/tutorials/models` for details about the format.
        """
        with torch.no_grad():  # https://github.com/sphinx-doc/sphinx/issues/4258
            input_lst = []
            for original_image in original_images:
                if self.input_format == "RGB":
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = self.pad.get_transform(image).apply_image(image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input = {"image": image, "height": height, "width": width}
                input_lst.append(input)
            predictions = self.model(input_lst)
            return predictions
        
class BatchPredictor:
    def __init__(self, cfg):
        self.cfg = cfg.clone()  # cfg can be modified by model
        self.model = build_model(self.cfg)
        self.model.eval()
        if len(cfg.DATASETS.TEST):
            self.metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

        checkpointer = DetectionCheckpointer(self.model)
        checkpointer.load(cfg.MODEL.WEIGHTS)

        self.aug = T.ResizeShortestEdge(
            [cfg.INPUT.MIN_SIZE_TEST, cfg.INPUT.MIN_SIZE_TEST], cfg.INPUT.MAX_SIZE_TEST
        )

        self.input_format = cfg.INPUT.FORMAT
        assert self.input_format in ["RGB", "BGR"], self.input_format

    def __call__(self, original_images: list):
        with torch.no_grad():
            input_lst = []
            for original_image in original_images:
                # Apply pre-processing to image.
                if self.input_format == "RGB":
                    # whether the model expects BGR inputs or RGB
                    original_image = original_image[:, :, ::-1]
                height, width = original_image.shape[:2]
                image = self.aug.get_transform(original_image).apply_image(original_image)
                image = torch.as_tensor(image.astype("float32").transpose(2, 0, 1))
                input = {"image": image, "height": height, "width": width}
                input_lst.append(input)
            predictions = self.model(input_lst)
            return predictions


def get_model(cfg, model_weights=None, instance_mode=ColorMode.IMAGE):
    return DeepSoloWrapper(cfg, model_weights, instance_mode)
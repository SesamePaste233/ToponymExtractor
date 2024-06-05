from PatchCutter.ImagePreprocessor import ImagePreprocessor
from Utils.visualizer import PolygonVisualizer

from PIL import Image

def pyramid_scan(img_path, output_path, spotter, num_layers = 2, save_visualization=False, save_results=True):
    image = Image.open(img_path)
    image_preprocessor = ImagePreprocessor(image, overlapping_tolerance=0.3, num_layers=num_layers, min_patch_resolution=384, max_patch_resolution=384)
    image_preprocessor.process()
    print("preprocessing done")
    all_layer_results = []

    base_image_batch, base_offset_xs, base_offset_ys = image_preprocessor.get_image_patches(0)
    vis = PolygonVisualizer()
    vis.canvas_from_patches(base_image_batch, base_offset_xs, base_offset_ys)

    for i in range(image_preprocessor.num_layers):
        # If you want to save for each layer, uncomment the following line
        # image_preprocessor.save_patches(os.path.join(output_dir, f'layer_{i}_patches'), layer=i)

        image_batch, offset_xs, offset_ys = image_preprocessor.get_image_patches(i)
        spotter.load_batch(image_batch, offset_xs, offset_ys)
        results = spotter.inference_batch(batch_size=8, rotations = [0])
        all_layer_results.extend(results)

    if save_visualization:
        vis.draw(all_layer_results).save(output_path.replace('.json', '.jpg'))
    if save_results:
        vis.save_json(all_layer_results, output_path)

    return all_layer_results
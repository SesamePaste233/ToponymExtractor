import os
from PIL import Image
import numpy as np

class ImagePreprocessor:
    def __init__(self, image: Image, overlapping_tolerance = 0.2, num_layers = 3, min_patch_resolution = 256, max_patch_resolution = 2048) -> None:
        self.image = image
        self.overlapping_tolerance = overlapping_tolerance
        self.num_layers = num_layers
        self.image_layers = [{} for _ in range(self.num_layers)]
        self.min_patch_resolution = min_patch_resolution
        self.max_patch_resolution = max_patch_resolution
        return
    
    # Crop the image into m times m patches and allow overlapping
    def crop(self, m: int):
        image_patches = []
        h, w = self.image.size
        
        if h > w:
            patch_size = int(h/(m - m * self.overlapping_tolerance + self.overlapping_tolerance))
            grid_size_h = int(patch_size * (1 - self.overlapping_tolerance))
            grid_size_w = (w - patch_size) // (m - 1)

        else:
            patch_size = int(w/(m - m * self.overlapping_tolerance + self.overlapping_tolerance))
            grid_size_w = int(patch_size * (1 - self.overlapping_tolerance))
            grid_size_h = (h - patch_size) // (m - 1)

        if grid_size_h * (m - 1) + patch_size >= h or grid_size_w * (m - 1) + patch_size >= w:
            patch_size -= 1

        h_offsets = [i * grid_size_h for i in range(m)]
        w_offsets = [i * grid_size_w for i in range(m)]
        for h in h_offsets:
            for w in w_offsets:
                patch = self.image.crop((h, w, h + patch_size, w + patch_size))
                image_patches.append({'image': patch, 'offset_x': h, 'offset_y': w})
        return image_patches
    
    def process(self, image: Image = None):
        if image == None:
            image = self.image
        
        h, w = self.image.size

        min_pixels = min(h, w)

        min_patches_m = int(min_pixels/(self.max_patch_resolution * (1 + self.overlapping_tolerance))) + 1

        max_patches_m = int(min_pixels/(self.min_patch_resolution * (1 - self.overlapping_tolerance))) + 1

        if min_patches_m < 3:
            min_patches_m = 3

        # Calculate the number of patches for each layer
        patches_m = np.linspace(max_patches_m, min_patches_m, self.num_layers, dtype = int)[::-1]
        print("Cropping patches:", patches_m)
        # Crop the image into patches
        for i, m in enumerate(patches_m):
            image_patches = self.crop(m)
            self.image_layers[i] = image_patches

        return self.image_layers

    def save_patches(self, folder_path = r'.', layer = 0, clear_folder = True):
        # Create folder if not exists
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        # Clear folder if required
        if clear_folder:
            for file in os.listdir(folder_path):
                os.remove(os.path.join(folder_path, file))

        for patch in self.image_layers[layer]:
            image = patch['image']
            offset_x = patch['offset_x']
            offset_y = patch['offset_y']
            image.save(os.path.join(folder_path, f'{layer}_{offset_y}_{offset_x}.png'))
        return
    
    def get_image_patches(self, layer):
        offset_xs = [p['offset_x'] for p in self.image_layers[layer]]
        offset_ys = [p['offset_y'] for p in self.image_layers[layer]]
        images = [p['image'] for p in self.image_layers[layer]]

        return images, offset_xs, offset_ys

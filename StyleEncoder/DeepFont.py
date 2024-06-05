import torch
from torch import nn

class DeepFontEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.Cu = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=11, stride=2), nn.ReLU(),       # output shape: 64 * 48 * 48
            nn.BatchNorm2d(64),             # output shape: 64 * 48 * 48
            nn.MaxPool2d(kernel_size=2),    # output shape: 64 * 24 * 24

            nn.Conv2d(64, 128, kernel_size=3, padding=1), nn.ReLU(),       # output shape: 128 * 24 * 24
            nn.BatchNorm2d(128),            # output shape: 128 * 24 * 24
            nn.MaxPool2d(kernel_size=2)     # output shape: 128 * 12 * 12
        ) # From Stacked Convolutional Auto-Encoder

        self.Cs = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), nn.ReLU(), # output shape: 256 * 12 * 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), # output shape: 256 * 12 * 12
            nn.Conv2d(256, 256, kernel_size=3, padding=1), nn.ReLU(), # output shape: 256 * 12 * 12

            nn.Flatten(),   # output shape: 36864

            nn.Linear(12 * 12 * 256, 4096), nn.ReLU(),  
            nn.Dropout(0.5),
            nn.Linear(4096, 4096), nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 128), nn.ReLU(),
            nn.Linear(128, 24)
        ) # From CNN


    def forward(self, X):
        X = self.Cu(X)
        X = self.Cs(X)

        return X


from PIL import Image
from typing import List, Union
import numpy as np

from tqdm import tqdm
import cv2

def _resize_and_pad(pil_img:Image.Image, target_size=(105, 105)):
    # Resize while preserving aspect ratio
    width, height = target_size
    if pil_img.width < pil_img.height:
        pil_img = pil_img.resize((int(pil_img.width * height / pil_img.height), height))
    else:
        pil_img = pil_img.resize((width, int(pil_img.height * width / pil_img.width)))
        
    img = np.array(pil_img)

    # Calculate the amount of padding needed
    top_pad = (target_size[1] - img.shape[0]) // 2
    bottom_pad = target_size[1] - img.shape[0] - top_pad
    left_pad = (target_size[0] - img.shape[1]) // 2
    right_pad = target_size[0] - img.shape[1] - left_pad

    # Pad the image with BORDER_REPLICATE
    img = cv2.copyMakeBorder(img, top_pad, bottom_pad, left_pad, right_pad, borderType=cv2.BORDER_CONSTANT, value=[0, 0, 0])

    return Image.fromarray(img)

def _img_to_tensor(pil_img: Union[Image.Image, List[Image.Image]]) -> torch.Tensor:
    '''Conver PIL image or list of PIL image into a tensor.
    '''
    if isinstance(pil_img, List):
        return torch.tensor([_img_to_tensor(img).cpu().detach().numpy() for img in pil_img])
    
    if pil_img.mode != 'L':
        pil_img = pil_img.convert('L')

    if pil_img.size[0] != 105 or pil_img.size[1] != 105:
        pil_img = _resize_and_pad(pil_img, (105, 105))

    img = torch.tensor(np.array(pil_img)).float()
    img = img / 255
    img = torch.unsqueeze(img, dim=0) # channel=1

    return img

def EncodeFontSingle(net: DeepFontEncoder, image_single: Image) -> 'list[float]':
    '''Encode a single font image into a list of features.
    '''

    X = _img_to_tensor(image_single).to(next(net.parameters()).device)

    with torch.no_grad():
        features = net(X)

    # Convert tensor to list
    features = features.tolist()

    return features

def EncodeFontSinglePath(net: DeepFontEncoder, image_path: str) -> 'list[float]':
    '''Encode a single font image into a list of features.
    '''
    img = Image.open(image_path)

    return EncodeFontSingle(net, img)

def EncodeFontBatch(net: DeepFontEncoder, image_batch: List[Image.Image]) -> 'list[list[float]]':
    '''Encode a batch of font images into a list of features.
    '''
    X = _img_to_tensor(image_batch).to(next(net.parameters()).device)

    with torch.no_grad():
        features = net(X)

    # Convert tensor to list
    features = features.tolist()

    return features

def EncodeFontBatchPath(net: DeepFontEncoder, image_path_lst: List[str], batch_size = 50) -> 'list[list[float]]':
    '''Encode a batch of font images into a list of features.
    '''

    # Split the image paths into batches
    image_path_batch = [image_path_lst[i:i + batch_size] for i in range(0, len(image_path_lst), batch_size)]

    feature_list = []
    for image_paths in tqdm(image_path_batch):
        image_batch = [Image.open(path) for path in image_paths]

        features = EncodeFontBatch(net, image_batch)
        feature_list.extend(features)

    return feature_list

def load_model(model_path: str, device = 'cpu') -> DeepFontEncoder:
    '''Load a model from a file.
    '''
    net = DeepFontEncoder()
    net.load_state_dict(torch.load(model_path))
    net.to(device)

    return net

def save_model(net: DeepFontEncoder, model_path: str) -> None:
    '''Save a model to a file.
    '''
    torch.save(net.state_dict(), model_path)

if __name__ == '__main__':
    deepfontencoder = load_model('dataset\models\DeepFontEncoder.pth')
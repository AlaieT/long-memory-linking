from PIL import Image
import torch
from torchvision import transforms


transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((108, 108))])


def get_image(path: str, params: list):
    left = params[0] - params[2]/2
    right = params[0] + params[2]/2
    top = params[1] - params[3]/2
    bottom = params[1] + params[3]/2

    img = transform(Image.open(path).crop((left, top, right, bottom)))
    img = img.unsqueeze(0).numpy()

    return img

import os
import argparse

import PIL.Image
from torchvision.models import ResNet50_Weights, resnet50

parser = argparse.ArgumentParser(description="recognize images based on Imagenet dataset")
parser.add_argument('-p', '--path', default='imgs',
                    type=str, help='path to directory of images')
args = parser.parse_args()

mypath = args.path
imgs = list()
if os.path.exists(mypath):
    for f in os.listdir(mypath):
        if os.path.isfile(os.path.join(mypath, f)) and f.endswith(('.png', '.jpg', '.jpeg')):
            imgs.append(f)

for f in imgs:
    # read image
    img = PIL.Image.open(os.path.join(mypath, f)).convert('RGB')

    # Initialize model with weights
    weights = ResNet50_Weights.DEFAULT
    model = resnet50(weights=weights)
    model.eval()

    # Initialize transforms
    preprocess = weights.transforms()

    # Apply transforms to image
    batch = preprocess(img).unsqueeze(0)

    # output prediction
    prediction = model(batch).squeeze(0).softmax(0)
    class_id = prediction.argmax().item()
    score = prediction[class_id].item()
    category_name = weights.meta["categories"][class_id]
    print(f"{category_name}: {100 * score:.1f}%")

import PIL.Image

from os import listdir
from os.path import isfile, join

from torchvision.models import resnet50, ResNet50_Weights

mypath = './imgs/'
imgs = [f for f in listdir(mypath) if isfile(join(mypath, f))]

for f in imgs:
    # read image
    img = PIL.Image.open(mypath + f).convert('RGB')

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

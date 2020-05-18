import torch

from model import BiSeNet
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
import cv2

n_classes = 19
net = BiSeNet(n_classes=n_classes)
net.load_state_dict(torch.load('./saved_models/mosaic.pth'))
net.eval()

to_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])
# Create the model and load the weights
#net = BiSeNet()
# Create dummy input
dummy_input = torch.rand(1, 3, 448, 448)

# Define input / output names
input_names = ["inputImage"]
output_names = ["outputImage"]
#print(net)
# Convert the PyTorch model to ONNX


with torch.no_grad():
    img = Image.open('./stata.jpg')
    image = img.resize((720, 720), Image.BILINEAR)
    img = to_tensor(image)
    img = torch.unsqueeze(img, 0)
    print(img.shape)
    out = net(img)
    #print(out)
    torch.onnx.export(net, img, "mosaic.onnx", verbose=False, image_input_names=input_names, image_ouput_names=output_names)
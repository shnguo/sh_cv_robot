import torch
from torchvision import transforms
from PIL import Image
from resnet.net import *

sample = Image.open('./花屏/data/test/0/微信图片_20230605161228.png')
tfms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])
sample = tfms(sample).unsqueeze(0)
print(sample.shape)
model = torch.load('花屏/model.pt')
model.eval()
nonlinear = nn.Softmax(dim=1)
outputs = nonlinear(model(sample))
print(outputs.tolist()[0])
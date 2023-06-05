import os
import clip
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn import Sequential, Module, Linear
import torchvision.models as models


class LASTED(nn.Module):
    def __init__(self, num_class=4):
        super().__init__()
        self.clip_model, self.preprocess = clip.load("RN50x64", device='cpu', jit=False)
        self.output_layer = Sequential(
            nn.Linear(1024, 1280),
            nn.GELU(),
            nn.Linear(1280, 512),
        )
        self.fc = nn.Linear(512, num_class)
        # self.text_input = clip.tokenize(['Real', 'Synthetic'])
        self.text_input = clip.tokenize(['Real Photo', 'Synthetic Photo', 'Real Painting', 'Synthetic Painting'])
        # self.text_input = clip.tokenize(['Real-Photo', 'Synthetic-Photo', 'Real-Painting', 'Synthetic-Painting'])
        # self.text_input = clip.tokenize(['a', 'b', 'c', 'd'])

    def forward(self, image_input, isTrain=True):
        if isTrain:
            logits_per_image, _ = self.clip_model(image_input, self.text_input.to(image_input.device))
            return None, logits_per_image
        else:
            image_feats = self.clip_model.encode_image(image_input)
            image_feats = image_feats / image_feats.norm(dim=1, keepdim=True)
            return None, image_feats

class ResNet(nn.Module):
    def __init__(self, num_class = 2):
        super(ResNet, self).__init__()
        self.resnet = models.resnet50(pretrained=True)
        # 将ResNet的所有层都加载进来
        self.features = nn.Sequential(*list(self.resnet.children())[:-1])
        # 将最后一个fc层去掉
        self.fc = nn.Linear(2048, num_class)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == '__main__':
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # model = LASTED().cuda()
    model = ResNet(num_classes=2)
    model.eval()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('Params: %.2f' % (params / (1024 ** 2)))

    x = torch.zeros([4, 3, 448, 448])
    # _, logits = model(x)
    outputs = model(x)
    print(outputs.shape)


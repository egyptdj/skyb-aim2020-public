import torch
import torch.nn as nn
from torchvision import models


class VGG19Loss(object):
    def __init__(self, layers, weights=None, loss_function=nn.MSELoss()):
        super(VGG19Loss, self).__init__()
        if isinstance(layers, str): layers = [layers]
        if weights==None: weights = [1.0]*len(layers)
        assert len(layers)==len(weights)
        vgg = models.vgg19(pretrained=True).features
        self.modules = nn.ModuleList()

        _modules = []
        _conv = 1
        _relu = 1
        _layer = 1
        for module in vgg.children():
            if isinstance(module, nn.Conv2d):
                name = f'conv{_layer}_{_conv}'
                _conv += 1
            elif isinstance(module, nn.ReLU):
                name = f'relu{_layer}_{_relu}'
                _relu += 1
            elif isinstance(module, nn.MaxPool2d):
                name = f'pool{_layer}'
                _conv = 1
                _relu = 1
                _layer += 1

            _modules.append(module)
            if name in layers:
                self.modules.append(nn.Sequential(*_modules))
                _modules = []
        self.weights = torch.FloatTensor(weights)
        self.loss_function = loss_function

        for param in self.modules.parameters():
            param.requires_grad = False

        self.mean = torch.FloatTensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
        self.stddev = torch.FloatTensor([0.229, 0.224, 0.225]).view(-1, 1, 1)


    def __call__(self, source, target):
        source = (source - self.mean) / self.stddev
        target = (target - self.mean) / self.stddev
        losses = []
        for weight, module in zip(self.weights, self.modules):
            source = module(source)
            target = module(target)
            losses.append(weight * self.loss_function(source, target))
        return torch.mean(torch.DoubleTensor(losses))


    def to(self, device):
        self.weights = self.weights.to(device)
        self.modules = self.modules.to(device)
        self.mean = self.mean.to(device)
        self.stddev = self.stddev.to(device)

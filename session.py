import random
import contextlib
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from loss import *
from metric import *
from utils.image import *
from IQA_pytorch import LPIPSvgg
from pytorch_msssim import ssim, MS_SSIM


class PyNetSession(object):
    def __init__(self, model, perceptual=False):
        self.model = model
        self.perceptual = perceptual
        self.optimizer = None
        self.scheduler = None
        self.criterions = None
        self.coefficients = None
        self.lpips = LPIPSvgg()
        self.loss = {'total': [], 'mse': [], 'vgg': [], 'msssim': []}
        self.metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        self.images = {'raw': None, 'enhanced': None, 'rgb': None}
        self.device = 'cpu'


    def set_optimizer(self, lr, level=None):
        assert level in [None, 0, 1, 2, 3, 4, 5], 'unknown level'
        if level is None:
            params = self.model.parameters()
        else:
            params = []
            for name, param in self.model.named_parameters():
                if (f'level{level}' in name) or ('conv1' in name):
                    params.append(param)
        self.optimizer = optim.Adam(params, lr=lr)


    def set_scheduler(self, lr, epochs, steps_per_epoch, pct_start=0.2, div_factor=2, final_div_factor=100):
        assert self.optimizer is not None, 'optimizer should be set before setting the scheduler'
        self.scheduler = optim.lr_scheduler.OneCycleLR(self.optimizer, lr*div_factor, epochs=epochs, steps_per_epoch=steps_per_epoch, pct_start=pct_start, div_factor=div_factor, final_div_factor=final_div_factor)


    def set_criterion(self, level=None):
        assert level in [0, 1, 2, 3, 4, 5], 'unknown level'
        criterions = [nn.MSELoss()]
        coefficients = [1.0]

        if level<4:
            perceptual_loss = VGG19Loss(['relu5_4'])
            perceptual_loss.to(self.device)
            criterions.append(perceptual_loss)
            coefficients.append(0.01)

        if level==0:
            criterions.append(MS_SSIM(data_range=1.0, size_average=True, channel=3))
            if self.perceptual:
                coefficients.append(-0.1)
            else:
                coefficients.append(-0.01)

        self.criterions = criterions
        self.coefficients = torch.FloatTensor(coefficients).to(self.device)


    def get_loss(self, empty_cache=True):
        loss = {'total': None, 'mse': None, 'vgg': None, 'msssim': None}
        for key in self.loss.keys():
            if len(self.loss[key])>0:
                loss[key] = np.mean(self.loss[key])
        if empty_cache: self.loss = {'total': [], 'mse': [], 'vgg': [], 'msssim': []}
        return loss


    def get_metrics(self, empty_cache=True):
        metrics = {'psnr': None, 'ssim': None, 'lpips': None}
        for key in self.metrics.keys():
            if len(self.metrics[key])>0:
                metrics[key] = np.mean(self.metrics[key])
        if empty_cache: self.metrics = {'psnr': [], 'ssim': [], 'lpips': []}
        return metrics


    def get_images(self):
        return self.images


    def step(self, data, level, train=False, augmentation=True):
        if train:
            assert self.optimizer is not None, 'optimizer should be set before training'
            self.model.train()
        else:
            self.model.eval()

        if self.criterions is None:
            self.set_criterion(level)

        with pass_context() if train else torch.no_grad():
            input = expand(data['raw'].to(self.device))
            target = expand(data['rgb'].to(self.device))

            if augmentation and train:
                k = random.randrange(8)
                input = augment(input, k)
                target = augment(target, k)
                enhanced = self.model(input, level)
            elif augmentation and not train:
                enhanced = torch.mean(torch.stack([augment(self.model(augment(input, k), level), k, inverse=True) for k in range(8)], dim=0), dim=0)
            else:
                enhanced = self.model(input, level)

            total_loss = 0.0
            for i, (coefficient, criterion) in enumerate(zip(self.coefficients, self.criterions)):
                if i==0:
                    mse_loss = coefficient * criterion(enhanced, target)
                    total_loss += mse_loss
                    self.loss['mse'].append(mse_loss.cpu().detach().numpy())
                elif i==1:
                    vgg_loss = coefficient * criterion(shrink(enhanced), shrink(target))
                    total_loss += vgg_loss
                    self.loss['vgg'].append(vgg_loss.cpu().detach().numpy())
                elif i==2:
                    msssim_loss = coefficient * criterion(shrink(enhanced), shrink(target))
                    total_loss += msssim_loss
                    self.loss['msssim'].append(msssim_loss.cpu().detach().numpy())

            if train:
                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()
                if self.scheduler is not None:
                    self.scheduler.step()

        input = shrink(input)
        enhanced = shrink(enhanced)
        target = shrink(target)

        self.loss['total'].append(total_loss.cpu().detach().numpy())
        self.metrics['psnr'].append(psnr(enhanced, target).detach().cpu().numpy())
        self.metrics['ssim'].append(ssim(enhanced, target, data_range=1.0, size_average=True).detach().cpu().numpy())
        if level==0: self.metrics['lpips'].append(self.lpips(enhanced, target).detach().cpu().numpy())
        self.images['raw'] = input.detach().cpu()[:,:3,:,:]
        self.images['enhanced'] = enhanced.detach().cpu()
        self.images['rgb'] = target.detach().cpu()

        torch.cuda.empty_cache()


    def to(self, device):
        self.model.to(device)
        self.lpips.to(device)
        self.device = device


    def parallel(self):
        self.model = nn.DataParallel(self.model)


@contextlib.contextmanager
def pass_context():
    yield None

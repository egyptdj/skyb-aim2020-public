import torch


def psnr(x1, x2, max_val=1.0):
    return 20*torch.log10(max_val/torch.sqrt(torch.mean((x1-x2)**2)))

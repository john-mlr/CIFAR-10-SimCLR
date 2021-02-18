import torch
import torchvision


import torch
import numpy as np
import torchvision

class SimCLRTransforms(object):
    def __init__(self, img_size, s=1):
        self.size = img_size
        self.blur_size = int(img_size[1] * .1)

        self.ColorDistort = torchvision.transforms.ColorJitter(brightness=0.8*s,
                                                                contrast=0.8*s,
                                                                saturation=0.8*s,
                                                                hue=0.2*s)

        if self.size % 2 == 0:
            self.GaussBlur = torchvision.transforms.GaussianBlur(kernel_size=self.blur_size+1,
                                                                sigma=(0.1, 2.0))
        else:
            self.GaussBlur = torchvision.transforms.GaussianBlur(kernel_size=self.blur_size,
                                                                 sigma=(0.1, 2.0))

        self.train_transform = torchvision.transforms.Compose([
                                                            torchvision.transforms.RandomResizedCrop(self.size),
                                                            torchvision.transforms.RandomHorizontalFlip(p=0.5),
                                                            torchvision.transforms.RandomApply([self.ColorDistort], p=0.8),
                                                            torchvision.transforms.RandomGrayscale(p=0.2),
                                                            torchvision.transforms.RandomApply([self.GaussBlur], p=0.5),
                                                            torchvision.transforms.ToTensor()
                                                            ])

        self.eval_transform = torchvision.transforms.Compose([
                                                            torchvision.transforms.ToTensor()
                                                            ])

    def __call__(self, img, pretraining):
        return self.train_transform(img)
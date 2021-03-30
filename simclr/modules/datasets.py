from torchvision import datasets

class SimCLRCIFAR10(datasets.CIFAR10):
    def __init__(self, download_path, train=True, transforms=None, pretraining=True):
        super().__init__(root=download_path, train=train, download=True)
        
        self.transforms = transforms
        self.pretraining = pretraining
        
    # no need for __len__ method, as it is inherited and unchanged from
    # parent.
        
    def __getitem__(self, idx):
        # retrieve images and labels using inherited get item.
        img, label = super.__getitem__(idx)
        
        # pretraining does not require labels, so does not return them.
        if self.pretraining:
            if self.transforms is not None:
                x_i = self.transforms(img)
                x_j = self.transforms(img)
                
                return (x_i, x_j)
            else:
                return (img, img)
        
        # pretraining=False indicates the dataset is for testing (linear eval), so return labels
        else:
            if self.transforms is not None:
                x_i = self.transforms(img)
                return (x_i, label)
            else:
                return (img, label) 
        
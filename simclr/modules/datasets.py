from torchvision import datasets

class SimCLRCIFAR10(datasets.CIFAR10):
    """ CIFAR-10 Dataset for SimCLR. If pretraining is selected, the set will return
        the same image twice with transformations applied. If it is not, the set 
        will return a transformed image and its label.
    
        Init args:
            download_path (str: path): path to store images locally on.
            train (bool: default True): If true, will draw imagesfrom the training split.
                                        False will draw from test.
            transforms (nn.Module or Object): transforms to apply to the images
            pretraining (bool: default True): Whether the dataset is for pretraining.
    """
    def __init__(self, download_path, train=True, transforms=None, pretraining=True):
        super().__init__(root=download_path, train=train, download=True)
        
        self.transforms = transforms
        self.pretraining = pretraining
        
    # no need for __len__ method, as it is inherited and unchanged from
    # parent.
        
    def __getitem__(self, idx):
        """ Returns a pair of images or an image and its label. Draws idx from
            the instances referenced DataLoader.
        """
        # retrieve images and labels using inherited __getitem__.
        img, label = super().__getitem__(idx)
        
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
        

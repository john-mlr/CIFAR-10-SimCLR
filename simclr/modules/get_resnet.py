import torchvision
import torch.nn as nn

def get_resnet(args, cifar=True):
    """ Generate resnet as the encoder model for SimCLR.
    
        Args:
            args: Important piece is args.encoder. Model specification.
                    Currently only support Resnets 50 and 18
                    
            cifar (bool): If true, will modify Resnet in accordance with the SimCLR paper. 
    """
    if args.encoder == "Resnet-50":
        model = torchvision.models.resnet50(pretrained=False)

        
    elif args.encoder == "Resnet-18":
        model = torchvision.models.resnet18(pretrained=False)
        
    if cifar:
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        
    model.fc = nn.Identity()
        
    return model
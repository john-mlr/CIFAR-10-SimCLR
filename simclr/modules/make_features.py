import torch

def make_features(loader, simclr_model, gpu):
    """ Make features from images by passing them through simclr encoder
    
        Args:
            loader (torch.DataLoader instance): Dataloader of images to be converted into features
            simclr_model (nn.Module): Trained SimCLR model to create features with.
            gpu (int): Rank of the gpu process is being performed on
    """
    
    feats = []
    labels = []
    for batch, (img, label) in enumerate(loader):
        img = img.cuda(gpu)
        label = label.cuda(gpu)
        
        with torch.no_grad():
            h_i, _, _, _ = simclr_model(img, img)
        
        feats.append(h_i)
        labels.append(label)
    
    feats = torch.cat(feats, dim=0)
    labels = torch.cat(labels, dim=0)
    
    
    return feats, labels

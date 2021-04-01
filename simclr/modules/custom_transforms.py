import torch
import numpy as np
from skimage import transform as sk_trans
from skimage import img_as_float32

"""
These transforms are customized transforms used for mammography analysis. As pytorch (and pil)
do not play nicely with 16-bit grayscale .pngs, some torchvision transforms needed to be implemented
in numpy and skimage. An additional highpass filter transform is also added, as we have trialed that
on the mammograms.
"""


def sk_loader(path):
    """Custom loader to work with 16-bit PNG images

    Args:
        path (string): path to image

    Dependencies:
        skimage.io, skimage.util
    """

    image = io.imread(path)
    seg_img = segment_breast(image)
    return img_as_float32(seg_img)

class HighPass(object):
    def __init__(self, img_shape, size_range=10):
        self.size_range = size_range
        self.img_shape = img_shape
        self.filt_list = self.gen_filt_list(self.size_range, self.img_shape)


    def gaussian_2d(self, x, y, mu1, mu2, sig):
        return np.exp(- (np.power(x - mu1, 2) + np.power(y - mu2, 2)) / (2 * np.power(sig, 2)))


    def gen_gaussian_2d_filter(self, size, radius):
        grid = np.zeros(size)
        for i in range(size[0]):
            for j in range(size[1]):
                grid[i,j] = 1 - (self.gaussian_2d(i, j, size[0]/2, size[1]/2, radius))
        return np.array(grid)

    def gen_filt_list(self, size_range, img_shape):
        filt_list = []
        for size in range(1, size_range, 1):
            filt_list.append(self.gen_gaussian_2d_filter(img_shape, np.sqrt(size)))
        return filt_list

    def filter_circle(self, filt, fft_img):
        filtered = np.multiply(filt, fft_img)
        return(filtered)

    def __call__(self, img):
        fft_img = np.fft.fftshift(np.fft.fft2(img))
        fft_highpass = self.filter_circle(random.choice(self.filt_list), fft_img)
        filt_img = np.fft.ifft2(np.fft.ifftshift(fft_highpass))
        return img_as_float32(np.abs(filt_img))
         

class ToTensor3D(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, image):
        new_shape = (3,) + image.shape
        dup_img = np.broadcast_to(image, new_shape)
        return torch.from_numpy(dup_img.copy())


class Resize(object):
    """Resize the image in a sample.

    Args:
        img_size (2-tuple of int): Desired image size.
    
    Dependencies:
        skimage
    """

    def __init__(self, image_size):
        assert isinstance(image_size, tuple)
        assert len(image_size) == 2
        self.image_size = image_size

    def __call__(self, image):
        return sk_trans.resize(image, self.image_size, 
                               mode='reflect', 
                               anti_aliasing=True)

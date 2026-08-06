import torch

class InterpolationMode:
    NEAREST = "nearest"
    NEAREST_EXACT = "nearest_exact"
    BILINEAR = "bilinear"
    BICUBIC = "bicubic"
    BOX = "box"
    HAMMING = "hamming"
    LANCZOS = "lanczos"

def resize(img, size, interpolation=InterpolationMode.BILINEAR, max_size=None, antialias=None):
    if isinstance(img, torch.Tensor):
        if img.ndim == 3:
            img_4d = img.unsqueeze(0)
            mode = "bilinear" if interpolation in [InterpolationMode.BILINEAR, "bilinear", InterpolationMode.BICUBIC, "bicubic"] else "nearest"
            res = torch.nn.functional.interpolate(img_4d, size=size, mode=mode)
            return res.squeeze(0)
        elif img.ndim == 4:
            mode = "bilinear" if interpolation in [InterpolationMode.BILINEAR, "bilinear", InterpolationMode.BICUBIC, "bicubic"] else "nearest"
            return torch.nn.functional.interpolate(img, size=size, mode=mode)
    return img

def pil_to_tensor(pic):
    import numpy as np
    if isinstance(pic, torch.Tensor):
        return pic
    img = torch.from_numpy(np.array(pic))
    if img.ndim == 2:
        return img.unsqueeze(0)
    return img.permute(2, 0, 1)

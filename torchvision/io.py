class ImageReadMode:
    UNCHANGED = 0
    GRAY = 1
    GRAY_ALPHA = 2
    RGB = 3
    RGB_ALPHA = 4

def decode_image(*args, **kwargs):
    raise NotImplementedError("decode_image mock")

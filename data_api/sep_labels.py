import numpy as np

def get_separate_labels(label_img):
    # 64-bit encoding
    dtype = label_img.dtype
    if dtype == np.uint8:
        w = 8
    elif dtype == np.uint16:
        w = 16
    else:
        raise Exception('Unknown dtype: "{}"'.format(dtype))
    l64 = label_img.astype('uint64')
    # Single channel mapping
    if len(l64.shape) == 3:
        l64i = ((l64[:, :, 0] << 2 * w) + (l64[:, :, 1] << w) + l64[:, :, 2])
    else:
        l64i = l64
    colors = np.unique(l64i)
    segmentations = []
    colors_all = []
    for c in colors:
        if c != 0:
            segmentation = (l64i == c).astype('uint8')
            segmentations.append(segmentation)
            colors_all.append(c)
    return segmentations, colors_all

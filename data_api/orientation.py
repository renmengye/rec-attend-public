import numpy as np

color_wheel = np.array([[255, 17, 0],
                        [255, 137, 0],
                        [230, 255, 0],
                        [34, 255, 0],
                        [0, 255, 213],
                        [0, 154, 255],
                        [9, 0, 255],
                        [255, 0, 255]], dtype='uint8')


def build_orientation_img(d, mask):
    """
    Args:
        d: [.., H, W, 8]
        mask: [..., H, W]
    """
    y = np.expand_dims(mask, -1)
    cw = color_wheel
    did = np.argmax(d, -1)
    new_shape = []
    for ss in range(len(y.shape) - 1):
        new_shape.append(y.shape[ss])
    new_shape.append(3)
    c2 = cw[did.reshape([-1])].reshape(new_shape)
    img = (c2 * y).astype('uint8')
    return img


def get_orientation(y, num_classes=8, encoding='one_hot'):
    """
    Args:
        y: [B, T, H, W]
    """
    # [H, 1]
    idx_y = np.arange(y.shape[2]).reshape([-1, 1])
    # [1, W]
    idx_x = np.arange(y.shape[3]).reshape([1, -1])
    # [H, W, 2]
    idx_map = np.zeros([y.shape[2], y.shape[3], 2])
    idx_map[:, :, 0] += idx_y
    idx_map[:, :, 1] += idx_x
    # [1, 1, H, W, 2]
    idx_map = idx_map.reshape([1, 1, y.shape[2], y.shape[3], 2])
    # [B, T, H, W, 1]
    y2 = np.expand_dims(y, 4)
    # [B, T, H, W, 2]
    y_map = idx_map * y2
    # [B, T, 1]
    y_sum = np.expand_dims(y.sum(axis=2).sum(axis=2), 3) + 1e-7
    # [B, T, 2]
    centroids = y_map.sum(axis=2).sum(axis=2) / y_sum
    # [B, T, 1, 1, 2]
    centroids = centroids.reshape([y.shape[0], y.shape[1], 1, 1, 2])
    # Orientation vector
    # [B, T, H, W, 2]
    ovec = (y_map - centroids) * y2
    # Normalize orientation [B, T, H, W, 2]
    ovec = (ovec + 1e-8) / \
        (np.sqrt((ovec * ovec).sum(axis=-1, keepdims=True)) + 1e-7)
    # [B, T, H, W]
    angle = np.arcsin(ovec[:, :, :, :, 0])
    xpos = (ovec[:, :, :, :, 1] > 0).astype('float')
    ypos = (ovec[:, :, :, :, 0] > 0).astype('float')
    # [B, T, H, W]
    angle = angle * xpos * ypos + (np.pi - angle) * (1 - xpos) * ypos + \
        angle * xpos * (1 - ypos) + \
        (-np.pi - angle) * (1 - xpos) * (1 - ypos)
    angle = angle + np.pi / 8
    # [B, T, H, W]
    angle_class = np.mod(
        np.floor((angle + np.pi) * num_classes / 2 / np.pi), num_classes)
    if encoding == 'one_hot':
        angle_class = np.expand_dims(angle_class, 4)
        clazz = np.arange(num_classes).reshape(
            [1, 1, 1, 1, -1])
        angle_one_hot = np.equal(angle_class, clazz).astype('float32')
        angle_one_hot = (angle_one_hot * y2).max(axis=1)
        return angle_one_hot.astype('uint8')
    elif encoding == 'class':
        # [B, H, W]
        return (angle_class * y).max(axis=1).astype('uint8')
    else:
        raise Exception('Unknown encoding type: {}'.format(encoding))

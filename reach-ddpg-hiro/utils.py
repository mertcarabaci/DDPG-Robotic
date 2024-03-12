import torch
import matplotlib.pyplot as plt
from constants import DEVICE


def plot_observation(observation):
    frame_count = observation.shape[0]
    _, axes = plt.subplots(1, frame_count, figsize=(frame_count*4, 5))
    for i in range(frame_count):
        axes[i].imshow(observation[i], cmap='gray')
        axes[i].axis('off')
    plt.show()


def show_img(img, hide_colorbar=False):
    if len(img.shape) < 3 or img.shape[2] == 1:
        plt.imshow(img, cmap='gray')
    else:
        plt.imshow(img)

    if not hide_colorbar:
        plt.colorbar()


def prepend_tuple(new_dim, some_shape):
    some_shape_list = list(some_shape)
    some_shape_list.insert(0, new_dim)
    return tuple(some_shape_list)


def replace_color(data, original, new_value):
    r1, g1, b1 = original
    r2, g2, b2 = new_value

    red, green, blue = data[:,:,0], data[:,:,1], data[:,:,2]
    mask = (red == r1) & (green == g1) & (blue == b1)
    data[:,:,:3][mask] = [r2, g2, b2]


def convert2tensor(array):
    if torch.is_tensor(array):
        return array
    return torch.tensor(array, dtype=torch.float32).to(DEVICE)
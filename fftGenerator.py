import numpy as np
from PIL import Image, ImageTk
from imageHandler import ImageHandler


def generate_fft(data: np.ndarray) -> np.ndarray:
    return np.fft.fft(data)


def fft_to_img(fft: np.ndarray) -> ImageTk.PhotoImage:
    return ImageTk.PhotoImage(Image.fromarray(fft.astype(np.uint8)))


def normalize(data: np.ndarray) -> np.ndarray:
    normalized_data = np.ndarray(data.shape)
    normalized_data.fill(0)
    max_data = data.max()
    min_data = data.min()
    for idx, value in enumerate(data):
        normal_value = (value - min_data) / (max_data - min_data) * 255
        normalized_data[idx] = normal_value
    return normalized_data

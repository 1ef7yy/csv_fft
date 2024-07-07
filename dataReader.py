import numpy as np

from PIL import Image, ImageTk


class DataReader:
    def __init__(self, filename: str):
        self.filename = filename

    def get_data(self) -> np.ndarray:
        return np.fromfile(self.filename, dtype=np.uint16)

    def reshape_arr(self, data: np.ndarray, height: int, width: int) -> np.ndarray:
        return data.reshape(data.shape[0] // height // width, height, width).astype(
            np.float32
        )

    def convert_bits(self, data: np.ndarray) -> np.ndarray:
        return np.divide(data, 256)

    def convert_to_pixels(self, data: np.ndarray) -> np.ndarray:
        return data[0]

    def arr_to_pil(self, img_arr: np.ndarray) -> Image:
        return Image.fromarray(img_arr)

    def resize(self, img: Image, width: int, height: int) -> None:
        img.resize((width, height), Image.ANTIALIAS)

    def get_pixels(self):
        return self.convert_to_pixels(
            self.convert_bits(self.reshape_arr(self.get_data(), 512, 640))
        )

    def convert_to_pixels(self, data: np.ndarray) -> np.ndarray:
        return data[0]

    def readFile(self) -> ImageTk.PhotoImage:
        data = self.get_data()

        reshaped = self.reshape_arr(data, 512, 640)

        converted = self.convert_bits(reshaped)

        pixels = self.convert_to_pixels(converted)

        return ImageTk.PhotoImage(self.arr_to_pil(pixels))

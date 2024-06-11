import numpy as np

from PIL import Image, ImageTk


class DataReader:
    def __init__(self, filename: str):
        self.filename = filename

    
    def _get_data(self):
        return np.fromfile(self.filename, dtype=np.uint16)
    

    def _reshape(self, data, height, width):
        return data.reshape(data.shape[0] // height // width, height, width).astype(np.float32)


    def _convert_bits(self, data):
        return np.divide(data, 256)
    
    def _convert_to_pixels(self, data):
        return data[0]


    def _arr_to_pil(self, img_arr):
        return Image.fromarray(img_arr)
        

    def resize(self, img, width, height):
        print(type(img))
        img.resize((width, height), Image.ANTIALIAS)

    def readFile(self):
        data = self._get_data()

        reshaped = self._reshape(data, 512, 640)

        converted = self._convert_bits(reshaped)

        pixels = self._convert_to_pixels(converted)

        return ImageTk.PhotoImage(self._arr_to_pil(pixels))
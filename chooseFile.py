import dataReader as dr

from fftGenerator import generate_fft, fft_to_img, normalize

from tkinter import filedialog

from imageHandler import *


def choose_file(src_img, fft_label):
    filename = filedialog.askopenfilename()

    if not filename:
        return
    reader = dr.DataReader(filename)
    img = reader.readFile()

    imagehandler = ImageHandler(img)
    imagehandler.resize_img(320, 256)

    img = imagehandler.get_img()

    src_img.configure(image=img)
    src_img.image = img

    fft = normalize(generate_fft(reader.get_pixels()))

    fft = fft_to_img(fft)

    ffthandler = ImageHandler(fft)
    ffthandler.resize_img(320, 256)

    fft = ffthandler.get_img()

    fft_label.configure(image=fft)
    fft_label.image = fft

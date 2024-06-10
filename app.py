import app_utils as utils

from tkinter import *
from tkinter import ttk
from tkinter import filedialog

from PIL import Image, ImageTk



root = Tk()

root.title("FFT normalizer")

frm = ttk.Frame(root, padding=10)


frm.grid()

img = Image.open("static/cat.jpg")
image = ImageTk.PhotoImage(img)






img_original = ttk.Label(frm, image=image)
fft_original = ttk.Label(frm, image=image)
eq = ttk.Label(frm, image=image)
img_processing = ttk.Label(frm, image=image)
fft_processing = ttk.Label(frm, image=image)
fft_mask = ttk.Label(frm, image=image)

img_original.grid(column=0, row=1)
fft_original.grid(column=1, row=1)
eq.grid(column=2, row=1)
img_processing.grid(column=0, row=3)
fft_processing.grid(column=1, row=3)
fft_mask.grid(column=2, row=3)



ttk.Button(frm, text="Choose File", command= lambda: utils.change_img(img_original)).grid(column=2, row=4)

root.mainloop()


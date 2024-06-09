from tkinter import filedialog
from PIL import ImageTk, Image


def change_img(label):
    file_path = filedialog.askopenfilename()
    if file_path:
        img = Image.open(file_path).resize((640, 512))
        img = ImageTk.PhotoImage(img)
        label.config(image=img)
        label.image = img

import sys
from PyQt5.QtWidgets import (
    QMainWindow,
    QPushButton,
    QSpinBox,
    QCheckBox,
    QFormLayout,
    QComboBox,
    QAction,
    QFileDialog,
    QApplication,
    QHBoxLayout,
    QWidget,
    QGridLayout,
    QVBoxLayout,
    QTabWidget,
    QLineEdit,
)
from PyQt5.QtCore import Qt
import numpy as np

import csvReader
import pyqtgraph as pg
from scipy import fftpack
import cv2 as cv
from scipy import interpolate
from numba import jit
import csv
from dataReader import DataReader
import widgets


class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.flag = False
        self.menu = QWidget(self)

        self.orig_img = widgets.OriginalImageWidget(self)
        self.fft_orig = widgets.OriginalFFTWidget(self)
        self.eqwidget = widgets.EQWidget(self)
        self.proc_img = widgets.ProcessingImage(self)
        self.proc_fft = widgets.ProcessingFFT(self)
        self.fft_mask = widgets.FFTMask(self)
        self.initUI()

    def initUI(self):
        self.menu.setGeometry(
            0, 0, self.frameGeometry().width(), self.frameGeometry().height()
        )
        self.createMenu()
        self.menu.show()
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)

        self.grid = QGridLayout()
        self.central_widget.setLayout(self.grid)

        self.grid.addWidget(self.orig_img.getWidget(), 0, 0)

        self.grid.addWidget(self.fft_orig.getWidget(), 0, 1)

        self.grid.addWidget(self.eqwidget.getWidget(), 0, 2)

        self.grid.addWidget(self.proc_img.getWidget(), 1, 0)

        self.grid.addWidget(self.proc_fft.getWidget(), 1, 1)

        self.grid.addWidget(self.fft_mask.getWidget(), 1, 2)


        # Make the grid scalable with size
        self.grid.setColumnStretch(0, 1)
        self.grid.setColumnStretch(1, 1)
        self.grid.setColumnStretch(2, 1)
        self.grid.setRowStretch(0, 1)
        self.grid.setRowStretch(1, 1)

        self.setDefaults()

        self.setWindowTitle("Program")
        self.setGeometry(100, 100, 1920, 1240)
        self.show()

    def createMenu(self):
        openFile = QAction("Open", self.menu)
        openFile.setShortcut("Ctrl+O")
        openFile.setStatusTip("Open new File")
        openFile.triggered.connect(self.showDialog)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu("&File")
        fileMenu.addAction(openFile)

    def showDialog(self):
        self.flag = True
        dialog = QFileDialog()
        dialog.setWindowFlag(Qt.WindowStaysOnTopHint)
        fname = dialog.getOpenFileName(
            dialog,
            "Open File",
            "",
            "(*.data *.csv *.jpg *.png)",
            options=QFileDialog.DontUseNativeDialog,
        )[0]
        if fname.endswith(".data"):
            self.filetype = "data"
            DR = DataReader(fname)
            data = DR.readFileNP()
        elif fname.endswith(".csv"):
            self.filetype = "csv"
            data = csvReader.read_csv(fname)
        else:
            self.filetype = "jpg"
            data = cv.imread(fname, 0)
        data = cv.resize(data, (640, 512))
        self.orig_img.updateImage(data, filetype=self.filetype)
        self.fft_orig.updateImage(data)


    def setDefaults(self): ...


if __name__ == "__main__":
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())

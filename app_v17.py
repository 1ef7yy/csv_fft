import sys
from PyQt5.QtWidgets import (QMainWindow, QPushButton, QDoubleSpinBox, QSpinBox, QCheckBox, QFormLayout, QComboBox,
    QAction, QFileDialog, QApplication, QHBoxLayout, QWidget, QGridLayout, QVBoxLayout, QTabWidget, QLayout, QLineEdit,
    QLabel)
from PyQt5.QtCore import Qt
import numpy as np
import pyqtgraph as pg
from scipy import signal, fftpack
import cv2 as cv
from scipy import interpolate
from numba import jit
import csv
import matplotlib.pyplot as plt

w = 320
h = 256
paddingTop = 20
paddingLeft = 20
framePadding = 30
menubarHeight = 25
leftSideWidth = w * 2 + framePadding + paddingLeft * 2 + 350
rightSideWidth = w + paddingLeft * 2 + 150
windowHeight = h * 2 + paddingTop * 2 + framePadding + 200
footerHeight = 30


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        self.flag = False
        self.menu = QWidget(self)
        self.workSpace = QWidget(self)
        self.leftTabs = QTabWidget(self.workSpace)
        self.imageProcessingTab = QWidget()
        self.videoProcessingTab = QWidget()
        self.leftTabs.addTab(self.imageProcessingTab, "Image")
        self.leftTabs.addTab(self.videoProcessingTab, "Video")
        self.rightTabs = QTabWidget(self.workSpace)
        self.radEqualizer = QWidget()
        self.vertEqualizer = QWidget()
        self.horEqualizer = QWidget()
        self.rightTabs.addTab(self.radEqualizer, "Radial")
        self.rightTabs.addTab(self.vertEqualizer, "Vertical")
        self.rightTabs.addTab(self.horEqualizer, "Horizontal")
        self.rightTabs.tabBarClicked.connect(self.changeEqualizer)
        self.footer = QWidget(self)
        self.aru_checkbox = QCheckBox('ARU', self.footer)
        self.saveProfileButton = QPushButton('Save profile', self.footer)
        self.loadProfileButton = QPushButton('Load profile', self.footer)

        self.ff_orig, self.ff_centr = 0, 0

        self.x_rad, self.x_vert, self.x_hor = np.linspace(0, 410, 10), \
                                              np.linspace(0, 320, 10), \
                                              np.linspace(0, 256, 10)
        self.y_rad, self.y_vert, self.y_hor = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), \
                                              np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]), \
                                              np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0])
        self.x = [self.x_rad, self.x_vert, self.x_hor]
        self.y = [self.y_rad, self.y_vert, self.y_hor]

        self.f_rad, self.f_vert, self.f_hor = self.createInterpolator(self.x_rad, self.y_rad), \
                                              self.createInterpolator(self.x_vert, self.y_vert), \
                                              self.createInterpolator(self.x_hor, self.y_hor)
        self.f = [self.f_rad, self.f_vert, self.f_hor]

        self.x_rad_plot, self.x_vert_plot, self.x_hor_plot = np.linspace(0, 410, 100), \
                                                             np.linspace(0, 320, 100), \
                                                             np.linspace(0, 256, 100)
        self.y_rad_plot, self.y_vert_plot, self.y_hor_plot = self.f_rad(self.x_rad_plot), \
                                                             self.f_vert(self.x_vert_plot), \
                                                             self.f_hor(self.x_hor_plot)
        self.x_plot = [self.x_rad_plot, self.x_vert_plot, self.x_hor_plot]
        self.y_plot = [self.y_rad_plot, self.y_vert_plot, self.y_hor_plot]

        self.X, self.Y = np.meshgrid(np.linspace(0, 320, 320), np.linspace(0, 256, 256))
        self.Z = np.zeros((256, 320))

        self.mask_rad = self.generateMask(self.f_rad, idx=0)
        self.mask_vert = self.generateMask(self.f_vert, idx=1)
        self.mask_hor = self.generateMask(self.f_hor, idx=2)
        self.masks = [self.mask_rad, self.mask_vert, self.mask_hor]
        self.mask = None

        self.img_orig = None
        self.img_orig_contrast = None
        self.img_orig_bright = None

        self.ff_centr = None
        self.ff_orig = None

        self.img_new = None
        self.img_new_current = None

        self.ff_new = None

        self.index = 0

        self.aru_cycles = 1
        self.brightness = 0

        self.settingsWindow = SettingsWindow(self)
        self.settingsWindow.aruCyclesField.setValue(self.aru_cycles)
        self.settingsWindow.brightnessField.setValue(self.brightness)

        self.filterWindow = FilterWindow(self)

        self.spinboxBar_rad, self.spinboxBar_vert, self.spinboxBar_hor = [], [], []
        for i in range(10):
            spinbox_rad = QSpinBox()
            spinbox_rad.setRange(0, 1000)
            spinbox_rad.setSingleStep(1)
            spinbox_rad.setValue(self.y_rad[i].astype(np.int32) * 100)
            spinbox_rad.setStyleSheet('color: black; background-color: white')
            spinbox_rad.valueChanged.connect(self.spinboxChanged)

            spinbox_vert = QSpinBox()
            spinbox_vert.setRange(0, 1000)
            spinbox_vert.setSingleStep(1)
            spinbox_vert.setValue(self.y_vert[i].astype(np.int32) * 100)
            spinbox_vert.setStyleSheet('color: black; background-color: white')
            spinbox_vert.valueChanged.connect(self.spinboxChanged)

            spinbox_hor = QSpinBox()
            spinbox_hor.setRange(0, 1000)
            spinbox_hor.setSingleStep(1)
            spinbox_hor.setValue(self.y_hor[i].astype(np.int32) * 100)
            spinbox_hor.setStyleSheet('color: black; background-color: white')
            spinbox_hor.valueChanged.connect(self.spinboxChanged)

            self.spinboxBar_rad.append(spinbox_rad)
            self.spinboxBar_vert.append(spinbox_vert)
            self.spinboxBar_hor.append(spinbox_hor)
        self.spinboxBar = [self.spinboxBar_rad, self.spinboxBar_vert, self.spinboxBar_hor]

        self.processingGrid = CustomGrid(self.imageProcessingTab, 2, 2)

        self.radialGrid = CustomGrid(self.radEqualizer, 2, 1)
        self.verticalGrid = CustomGrid(self.vertEqualizer, 2, 1)
        self.horizontalGrid = CustomGrid(self.horEqualizer, 2, 1)
        self.equalizerGrids = [self.radialGrid, self.verticalGrid, self.horizontalGrid]

        self.initUI()

    def initUI(self):
        self.menu.setGeometry(0, 0, leftSideWidth + rightSideWidth, menubarHeight)
        self.createMenu()
        self.menu.show()
        self.workSpace.setGeometry(0, menubarHeight, leftSideWidth + rightSideWidth, windowHeight)
        self.workSpace.setStyleSheet('background-color: gray')
        self.workSpace.show()
        self.leftTabs.setGeometry(0, 5, leftSideWidth, windowHeight)
        self.leftTabs.show()
        self.rightTabs.setGeometry(leftSideWidth, 5, rightSideWidth, windowHeight)
        self.rightTabs.show()
        self.footer.setGeometry(0, menubarHeight + windowHeight, leftSideWidth + rightSideWidth, footerHeight)
        self.footer.setStyleSheet('background-color: gray')
        self.aru_checkbox.move(10, 0)
        self.aru_checkbox.setStyleSheet('color: white')
        self.saveProfileButton.move(leftSideWidth + rightSideWidth - 90, 0)
        self.saveProfileButton.setStyleSheet('background-color: white')
        self.saveProfileButton.clicked.connect(self.saveProfile)
        self.loadProfileButton.move(leftSideWidth + rightSideWidth - 180, 0)
        self.loadProfileButton.setStyleSheet('background-color: white')
        self.loadProfileButton.clicked.connect(self.loadProfile)
        self.aru_checkbox.stateChanged.connect(self.isChecked)
        self.setDefault()
        self.setGeometry(0, 0, leftSideWidth + rightSideWidth, windowHeight + footerHeight + menubarHeight)
        self.setWindowTitle('Program')
        self.show()

    def showDialog(self):
        self.flag = True
        dialog = QFileDialog()
        dialog.setWindowFlag(Qt.WindowStaysOnTopHint)
        fname = dialog.getOpenFileName(dialog, 'Open File', '', '(*.data *.jpg *.png)', options=QFileDialog.DontUseNativeDialog)[0]
        if fname[-5:] == ".data":
            hw = h * w
            data = np.fromfile(fname, dtype=np.uint16)[0:hw].reshape(h, w)
            data = (data // 256).astype(np.uint8)
        else:
            data = cv.imread(fname, 0)
        data = cv.resize(data, (640, 512))

        self.img_orig = data
        self.ff_orig, self.ff_centr = self.FFT(self.img_orig)
        self.ff_centr = self.toImage(self.ff_centr)
        self.processingGrid.updateWidget(0, 0, self.img_orig, mode="image")
        self.processingGrid.updateWidget(0, 1, self.ff_centr, mode="image")

        self.processing()

        self.settingsWindow.aru_cycles = self.aru_cycles
        self.settingsWindow.aruCyclesField.setValue(self.aru_cycles)

    def spinboxChanged(self):
        for i in range(10):
            val = self.spinboxBar[self.index][i].value()
            self.y[self.index][i] = val / 100
        self.f[self.index] = self.createInterpolator(self.x[self.index], self.y[self.index])
        self.masks[self.index] = self.generateMask(self.f[self.index], self.index)
        self.y_plot[self.index] = self.f[self.index](self.x_plot[self.index])
        self.y_plot[self.index] = self.setZeroLimit(self.y_plot[self.index])
        self.equalizerGrids[self.index].updateWidget(0, 0, x=self.x_plot[self.index], y=self.y_plot[self.index],
                                                     x_dots=self.x[self.index], y_dots=self.y[self.index],
                                     spinboxBar=self.spinboxBar[self.index])
        self.equalizerGrids[self.index].updateWidget(1, 0, self.masks[self.index], mode="image")
        if self.flag == False:
            return
        self.processing()

    def processing(self):
        self.mask = self.masks[0] * self.masks[1] * self.masks[2]
        ff_new = fftpack.ifftshift(self.mask) * self.ff_orig
        self.img_new = np.round(np.real(fftpack.ifft2(ff_new)))
        self.img_new = self.img_new / self.img_new.mean() * self.img_orig.mean()
        #self.img_new = ((self.img_new - self.img_new.min()) * 255 / (self.img_new.max()- self.img_new.min())).astype(np.uint8)
        #print(self.img_new.min(), self.img_new.max())
        if self.aru_checkbox.isChecked():
            self.img_new_current = ARU(self.img_new).processing(self.aru_cycles)
        else:
            self.img_new_current = self.toImage(self.img_new, self.brightness)
        self.drawProcessingPair()

    def drawProcessingPair(self):
        _, ff = self.FFT(self.img_new_current)
        ff = self.toImage(ff)
        self.processingGrid.updateWidget(1, 0, self.img_new_current, mode="image")
        self.processingGrid.updateWidget(1, 1, ff, mode="image")
        if self.settingsWindow.histFlag:
            self.settingsWindow.histProcessing(self.img_new_current, 1, 0)
        if self.filterWindow.flag:
            self.filterWindow.processing()

    @staticmethod
    @jit(nopython=True)
    def toImage(img, b=0):
        img = img.astype(np.int32)
        img_out = np.zeros_like(img)
        for i in range(512):
            for j in range(640):
                elem = img[i][j] + b
                if elem <= 0:
                    img_out[i][j] = 0
                elif elem >= 255:
                    img_out[i][j] = 255
                else:
                    img_out[i][j] = elem
        return img_out.astype(np.uint8)

    @staticmethod
    @jit(nopython=True)
    def setZeroLimit(array):
        for i in range(len(array)):
            if array[i] < 0:
                array[i] = 0
        return array

    def FFT(self, img):
        fft_orig = fftpack.fft2(img)
        ff_centr = np.log(np.abs(fftpack.fftshift(fft_orig)) + 1)
        return fft_orig, ff_centr

    def generateMask(self, f, idx):
        if idx == 0:
            self.Z = f((self.X ** 2 + self.Y ** 2) ** (1 / 2))
        elif idx == 1:
            self.Z = f(self.X)
        else:
            self.Z = f(self.Y)

        self.Z = np.array([[0 if self.Z[i][j] < 0 else self.Z[i][j] for j in range(self.Z.shape[1])] for i in
                           range(self.Z.shape[0])])
        Z1 = self.Z[:, ::-1]
        mask_half = np.concatenate((Z1, self.Z), 1)
        mask = np.concatenate((mask_half[::-1, :], mask_half), 0)
        return mask

    def createInterpolator(self, x, y):
        return interpolate.Akima1DInterpolator(x, y)

    def createMenu(self):
        openFile = QAction('Open', self.menu)
        openFile.setShortcut('Ctrl+O')
        openFile.setStatusTip('Open new File')
        openFile.triggered.connect(self.showDialog)

        openSettings = QAction('Settings', self.menu)
        openSettings.triggered.connect(self.settingsWindow.show)

        openFilter = QAction('Filter', self.menu)
        openFilter.triggered.connect(self.filterWindow.show)

        menubar = self.menuBar()
        fileMenu = menubar.addMenu('&File')
        fileMenu.addAction(openFile)
        menubar.addAction(openSettings)
        menubar.addAction(openFilter)

    def changeEqualizer(self, index):
        self.index = index
        if self.flag == False:
            return
        self.processing()

    def isChecked(self, state):
        if self.flag == False:
            return
        if state == Qt.Checked:
            self.img_new_current = ARU(self.img_new).processing(self.aru_cycles)
        else:
            self.img_new_current = self.toImage(self.img_new, self.brightness)
        self.drawProcessingPair()

    def saveProfile(self):
        dialog = QFileDialog()
        dialog.setWindowFlag(Qt.WindowStaysOnTopHint)
        fname = dialog.getSaveFileName(dialog, 'Save File', '', 'CSV (*.csv)', options=QFileDialog.DontUseNativeDialog)[0]
        print(fname)
        if fname[-4:] != '.csv':
            fname += '.csv'
        with open(fname, 'w', newline='') as File:
            writer = csv.writer(File)
            writer.writerows([self.y_rad, self.y_vert, self.y_hor])
        print(fname)
        cv.imwrite("fft_transform.png", self.img_new)
        print()

    def loadProfile(self):
        dialog = QFileDialog()
        dialog.setWindowFlag(Qt.WindowStaysOnTopHint)
        fname = dialog.getOpenFileName(dialog, 'Open file', '', 'CSV (*.csv)', options=QFileDialog.DontUseNativeDialog)[0]
        rows = []
        with open(fname, 'r', newline='') as File:
            reader = csv.reader(File)
            for row in reader:
                rows.append(row)
        for j in range(3):
            arr = np.array(rows[j]).astype(np.float16)
            for i in range(10):
                self.spinboxBar[j][i].setValue(arr[i] * 100)
        ind = self.index
        for i in range(3):
            self.index = i
            self.spinboxChanged()
        self.index = ind
        print(fname)

    def acceptSettings(self):
        self.aru_cycles = self.settingsWindow.aru_cycles
        self.brightness = self.settingsWindow.brightness
        if self.flag == False:
            return
        self.processing()

    def setDefault(self):
        zeroes = np.array([[129 for _ in range(w)] for __ in range(h)])

        self.processingGrid.addNewWidget(0, 0, pg.PlotWidget(), "Original Image")
        self.processingGrid.updateWidget(0, 0, zeroes, mode="image")

        self.processingGrid.addNewWidget(0, 1, pg.PlotWidget(), "Original FFT")
        self.processingGrid.updateWidget(0, 1, zeroes, mode="image")

        self.processingGrid.addNewWidget(1, 0, pg.PlotWidget(), "Processing Image")
        self.processingGrid.updateWidget(1, 0, zeroes, mode="image")

        self.processingGrid.addNewWidget(1, 1, pg.PlotWidget(), "Processing FFT")
        self.processingGrid.updateWidget(1, 1, zeroes, mode="image")

        self.radialGrid.addNewWidget(0, 0, pg.PlotWidget(), "Radial Equalizer Curve")
        self.radialGrid.updateWidget(0, 0, x=self.x_rad_plot, y=self.y_rad_plot, x_dots=self.x_rad, y_dots=self.y_rad,
                                     spinboxBar=self.spinboxBar_rad)

        self.verticalGrid.addNewWidget(0, 0, pg.PlotWidget(), "Vertical Equalizer Curve")
        self.verticalGrid.updateWidget(0, 0, x=self.x_vert_plot, y=self.y_vert_plot, x_dots=self.x_vert, y_dots=self.y_vert,
                                     spinboxBar=self.spinboxBar_vert)

        self.horizontalGrid.addNewWidget(0, 0, pg.PlotWidget(), "Horizontal Equalizer Curve")
        self.horizontalGrid.updateWidget(0, 0, x=self.x_hor_plot, y=self.y_hor_plot, x_dots=self.x_hor, y_dots=self.y_hor,
                                     spinboxBar=self.spinboxBar_hor)

        self.radialGrid.addNewWidget(1, 0, pg.PlotWidget(), "FFT Mask")
        self.radialGrid.updateWidget(1, 0, self.mask_rad, mode="image")

        self.verticalGrid.addNewWidget(1, 0, pg.PlotWidget(), "FFT Mask")
        self.verticalGrid.updateWidget(1, 0, self.mask_vert, mode="image")

        self.horizontalGrid.addNewWidget(1, 0, pg.PlotWidget(), "FFT Mask")
        self.horizontalGrid.updateWidget(1, 0, self.mask_hor, mode="image")

    def closeEvent(self, event):
        app.quit()


class CustomGrid(QGridLayout):
    def __init__(self, parent=None, rows=1, cols=1):
        super(CustomGrid, self).__init__(parent)
        self.widgets = []
        for i in range(rows):
            r = []
            for j in range(cols):
                w = QWidget()
                r.append(w)
                self.addWidget(w, i, j)
            self.widgets.append(r)

    def addNewWidget(self, i, j, widget, title):
        self.widgets[i][j] = widget
        self.widgets[i][j].setTitle(title)
        self.addWidget(widget, i, j)

    def updateWidget(self, i, j, data=None, mode='plot', x=None, y=None, x_dots=None, y_dots=None, spinboxBar=None):
        if mode == "image":
            self.removeWidget(self.widgets[i][j])
            self.widgets[i][j].clear()
            self.widgets[i][j].invertY(True)
            #self.widgets[i][j].addItem(pg.ImageItem(data.T, levels=[0, 255]))
            self.widgets[i][j].addItem(pg.ImageItem(data.T))
            self.addWidget(self.widgets[i][j], i, j)
        elif mode == "plot":
            vbox = QVBoxLayout()
            self.removeWidget(self.widgets[i][j])
            self.widgets[i][j].clear()
            self.widgets[i][j].showGrid(x=True, y=True)
            self.widgets[i][j].plot(x, y)
            self.widgets[i][j].plot(x_dots, y_dots, pen=None, symbol='o')
            hbox = QHBoxLayout()
            for k in range(10):
                spinboxBar[k].setValue(np.round(y_dots[k].astype(np.int32) * 100))
                hbox.addWidget(spinboxBar[k])
            vbox.addWidget(self.widgets[i][j])
            vbox.addLayout(hbox)
            self.addLayout(vbox, i, j)
        elif mode == 'hist':
            self.removeWidget(self.widgets[i][j])
            self.widgets[i][j].clear()
            self.widgets[i][j].plot(x, y)
            self.addWidget(self.widgets[i][j], i, j)


class ARU:
    def __init__(self, image):
        self.processing_image = image

    def processing(self, cycles):
        k, b = 1, self.processing_image.mean()
        for _ in range(cycles):
            k, b, self.processing_image = self.main_func(self.processing_image, k, b)
        return self.processing_image.astype(np.uint8)

    @staticmethod
    @jit(nopython=True)
    def main_func(image, k, b, x=0.015, y=0.03, N=512*640):
        N_black, N_white = 0, 0
        processing_image = np.zeros_like(image)
        s = 0
        for i in range(512):
            for j in range(640):
                elem = (image[i][j] - b) * k + 96
                s += elem
                if elem <= 0:
                    elem = 0
                    N_black += 1
                elif elem >= 255:
                    elem = 255
                    N_white += 1
                processing_image[i][j] = elem
        if (N_black + N_white) <= x * N:
            k = k * (1 + 1 / 32) + 1e-5
        elif (N_black + N_white) >= y * N:
            k = k * (1 - 1 / 32) - 1e-5
        b = s / N
        return k, b, processing_image.astype(np.uint8)


class SettingsWindow(QWidget):
    def __init__(self, parentWindow):
        super().__init__()
        self.parentWindow = parentWindow
        self.histFlag = False
        self.aru_cycles = None
        self.brightness = None

        self.footer = QWidget(self)
        self.footer.setGeometry(0, 270, 500, 30)

        self.histWindow = QWidget()
        self.histWindow.setGeometry(100, 150, 400, 500)
        self.histWindow.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.histWindow.setWindowTitle('Histograms')
        self.grid = CustomGrid(self.histWindow, 2, 1)
        self.x = np.arange(256)
        self.original_hist = np.zeros_like(self.x)
        self.processing_hist = np.zeros_like(self.x)
        self.grid.addNewWidget(0, 0, pg.PlotWidget(), 'Original Hist')
        self.grid.addNewWidget(1, 0, pg.PlotWidget(), 'Processing Hist')

        self.aruCyclesField = QSpinBox()
        self.aruCyclesField.setRange(1, 30)
        self.aruCyclesField.valueChanged.connect(self.change)
        self.brightnessField = QSpinBox()
        self.brightnessField.setRange(-255, 255)
        self.brightnessField.valueChanged.connect(self.change)
        flo = QFormLayout(self)
        flo.addRow("ARU Cycles", self.aruCyclesField)
        flo.addRow("Brightness", self.brightnessField)

        self.closeButton = QPushButton('Close', self.footer)
        self.closeButton.move(420, 3)
        self.closeButton.clicked.connect(self.onCloseButton)
        self.histButton = QPushButton('Histograms', self.footer)
        self.histButton.move(5, 3)
        self.histButton.clicked.connect(self.onHistButton)

        self.setWindowTitle('Settings')
        self.move(680, 300)
        self.setFixedSize(500, 300)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    def change(self):
        self.aru_cycles = self.aruCyclesField.value()
        self.brightness = self.brightnessField.value()
        self.parentWindow.acceptSettings()
        if self.parentWindow.filterWindow.flag:
            self.parentWindow.filterWindow.processing()

    def onHistButton(self):
        self.histFlag = True
        self.histWindow.show()
        if self.parentWindow.flag == False:
            return
        self.histProcessing(self.parentWindow.img_orig, 0, 0)
        self.histProcessing(self.parentWindow.img_new_current, 1, 0)

    def onCloseButton(self):
        self.histFlag = False
        self.histWindow.close()
        self.close()

    @staticmethod
    @jit(nopython=True)
    def calculateHists(img):
        hist = np.zeros(256)
        for i in range(512):
            for j in range(640):
                idx = img[i][j]
                hist[idx] += 1
        return hist

    def histProcessing(self, img, i, j):
        hist = self.calculateHists(img.astype(np.int32))
        self.grid.updateWidget(i, j, x=self.x, y=hist, mode="hist")


class FilterWindow(QWidget):
    def __init__(self, parentWindow):
        super().__init__()
        self.parentWindow = parentWindow
        self.img = None
        self.img_processed = None
        self.flag = False
        self.filter = None
        self.denominator = None
        self.filter_fpga = None
        self.algorithm_flag = 1

        self.imgWindow = QWidget(self)
        self.imgWindow.setGeometry(0, 0, 620, 450)
        self.imgWindow.setWindowFlag(Qt.WindowStaysOnTopHint)
        self.imgWindow.setWindowTitle('Filtered Image')
        self.grid = CustomGrid(self.imgWindow, 1, 1)
        self.grid.addNewWidget(0, 0, pg.PlotWidget(), 'Filtered Image')
        zeroes = np.array([[129 for _ in range(w)] for __ in range(h)])
        self.grid.updateWidget(0, 0, zeroes, mode="image")

        self.footer = QWidget(self)
        self.footer.setGeometry(0, 450, 620, 200)

        self.algorithm = QComboBox()
        self.algorithm.addItems(["Algorithm 1", "Algorithm 2"])
        self.algorithm.activated[str].connect(self.changeAlgorithm)
        self.rowField = QSpinBox()
        self.rowField.setRange(1, 100)
        self.rowField.setSingleStep(2)
        self.colField = QSpinBox()
        self.colField.setRange(1, 100)
        self.colField.setSingleStep(2)
        self.flo = QFormLayout(self.footer)
        self.flo.addRow("Algorithm", self.algorithm)
        self.flo.addRow("Row", self.rowField)
        self.flo.addRow("Col", self.colField)

        self.closeButton = QPushButton('Close', self.footer)
        self.closeButton.setGeometry(510, 95, 100, 25)
        self.closeButton.clicked.connect(self.onCloseButton)
        self.okButton = QPushButton('Ok', self.footer)
        self.okButton.setGeometry(10, 95, 100, 25)
        self.okButton.clicked.connect(self.onOkButton)
        self.denominatorComboBox = QComboBox(self.footer)
        values = [str(2 ** i) for i in range(1, 13)]
        self.denominatorComboBox.addItems(values)
        self.denominatorComboBox.setStyleSheet('color: black; background-color: white')
        self.denominatorComboBox.setGeometry(120, 95, 50, 25)

        '''self.denominatorSpinBox = QSpinBox(self.footer)
        self.denominatorSpinBox.setRange(2, 1024)
        self.denominatorSpinBox.setStyleSheet('color: black; background-color: white')
        self.denominatorSpinBox.setGeometry(120, 95, 50, 25)'''

        self.setWindowTitle('Filter')
        self.move(600, 50)
        self.setFixedSize(620, 580)
        self.setWindowFlag(Qt.WindowStaysOnTopHint)

    def showFilter(self):
        if self.filter is not None:
            self.filterWindow = QWidget()
            self.filterWindow.setWindowFlag(Qt.WindowStaysOnTopHint)
            self.filterWindow.setWindowTitle('FPGA Filter')

            grid = QGridLayout()
            self.filterWindow.setLayout(grid)
            positions = [(i, j) for i in range(self.filter_fpga.shape[0]) for j in range(self.filter_fpga.shape[1])]
            for position in positions:
                cell = QLineEdit(str(int(self.filter_fpga[position[0]][position[1]])))
                cell.setFixedWidth(50)
                cell.setReadOnly(True)
                grid.addWidget(cell, *position)

            dinomimatorField = QLineEdit(str(self.denominator))
            dinomimatorField.setReadOnly(True)
            sumField = QLineEdit(str(np.round(np.sum(self.filter_fpga / self.denominator), 5)))
            sumField.setReadOnly(True)
            footer = QWidget()
            flo = QFormLayout(footer)
            flo.addRow("Denominator", dinomimatorField)
            flo.addRow("Sum Coefficients", sumField)
            grid.addWidget(footer, positions[-1][0] + 1, 0, positions[-1][0] + 1, positions[-1][-1])

            self.filterWindow.move(160, 180)
            self.filterWindow.show()

    def FPGA_filter(self, filter, n):
        return np.round(filter * n)

    def changeAlgorithm(self, text):
        if text == "Algorithm 2" and self.algorithm_flag == 1:
            self.flo.removeRow(2)
            self.flo.removeRow(1)
            self.sizeField = QSpinBox()
            self.sizeField.setRange(3, 100)
            self.sizeField.setSingleStep(2)
            self.flo.addRow("Size", self.sizeField)
            self.algorithm_flag = 2
        if text == "Algorithm 1" and self.algorithm_flag == 2:
            self.rowField = QSpinBox()
            self.rowField.setRange(1, 100)
            self.rowField.setSingleStep(2)
            self.colField = QSpinBox()
            self.colField.setRange(1, 100)
            self.colField.setSingleStep(2)
            self.flo.removeRow(1)
            self.flo.addRow("Row", self.rowField)
            self.flo.addRow("Col", self.colField)
            self.algorithm_flag = 1

    def processing(self):
        if self.parentWindow.aru_checkbox.isChecked():
            self.img_processed = ARU(self.img).processing(self.parentWindow.aru_cycles)
        else:
            self.img_processed = self.parentWindow.toImage(self.img, self.parentWindow.brightness)
        self.showImg()

    def onCloseButton(self):
        self.imgWindow.close()
        self.close()
        self.flag = False

    def onOkButton(self):
        if self.parentWindow.flag == False:
            return
        self.flag = True

        if self.algorithm.currentText() == "Algorithm 1":
            n = int(self.rowField.value())
            m = int(self.colField.value())
            self.filter = self.Algorithm1(n, m, self.parentWindow.mask)
        else:
            n = int(self.sizeField.value())
            self.filter = self.Algorithm2(n)

        filter = self.filter / np.sum(self.filter)
        self.denominator = int(self.denominatorComboBox.currentText())
        self.filter_fpga = self.FPGA_filter(filter, self.denominator)
        self.img = cv.filter2D(self.parentWindow.img_orig, -1, self.filter_fpga / self.denominator)
        print(np.sum(self.filter_fpga / self.denominator))
        if self.parentWindow.aru_checkbox.isChecked():
            self.img_processed = ARU(self.img).processing(self.parentWindow.aru_cycles)
        else:
            self.img_processed = self.parentWindow.toImage(self.img, self.parentWindow.brightness)
        self.showImg()
        self.showFilter()

    def showImg(self):
        self.grid.updateWidget(0, 0, data=self.img_processed, mode='image')
        self.imgWindow.show()

    def Algorithm1(self, n, m, mask):
        mask = np.real(fftpack.ifft2(fftpack.ifftshift(mask)))
        mask = fftpack.fftshift(mask)
        filter = mask[256 - n // 2: 257 + n // 2, 320 - m // 2: 321 + m // 2]
        print(filter / np.sum(filter))
        return np.array(filter)

    def Algorithm2(self, n):
        m = np.ceil((n // 2 + 1) * np.sqrt(2)).astype(np.int16)
        x1 = (np.arange(m)).astype(np.int16)
        x2 = (np.arange(n // 2 + 1)).astype(np.int16)
        X, Y = np.meshgrid(np.arange(0, n // 2 + 1), np.arange(0, n // 2 + 1))

        x_plot_rad = np.linspace(0, 410, 410)
        y_plot_rad = self.parentWindow.f[0](x_plot_rad)
        i = np.round(np.linspace(0, 409, m)).astype(np.int16)
        y_rad = y_plot_rad[i]
        f_rad = self.parentWindow.createInterpolator(x1, y_rad)
        Z_rad = f_rad((X ** 2 + Y ** 2) ** (1 / 2))

        x_plot_vert = np.linspace(0, 320, 320)
        y_plot_vert = self.parentWindow.f[1](x_plot_vert)
        i = np.round(np.linspace(0, 319, n // 2 + 1)).astype(np.int16)
        y_vert = y_plot_vert[i]
        f_vert = self.parentWindow.createInterpolator(x2, y_vert)
        Z_vert = f_vert(X)

        x_plot_hor = np.linspace(0, 256, 256)
        y_plot_hor = self.parentWindow.f[2](x_plot_hor)
        i = np.round(np.linspace(0, 255, n // 2 + 1)).astype(np.int16)
        y_hor = y_plot_hor[i]
        f_hor = self.parentWindow.createInterpolator(x2, y_hor)
        Z_hor = f_hor(Y)

        Z = Z_rad * Z_vert * Z_hor
        Z1 = Z[:, ::-1]
        mask_half = np.concatenate((Z1[:, :-1], Z), 1)
        mask = np.concatenate((mask_half[::-1, :][:-1, :], mask_half), 0)

        filter = np.real(fftpack.ifft2(fftpack.ifftshift(mask)))
        filter = fftpack.fftshift(filter)
        print(filter / np.sum(filter))

        return filter


if __name__ == '__main__':
    app = QApplication(sys.argv)
    ex = MainWindow()
    sys.exit(app.exec_())


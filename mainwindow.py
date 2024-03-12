# This Python file uses the following encoding: utf-8
import sys
import os
#from PyQt5 import QtCore, QtGui, QtWidgets
#from PyQt5.QtWidgets import QWidget, QFileDialog, QMainWindow
#from PyQt5.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow
import numpy as np
import cv2
import pytesseract
import matplotlib.pyplot as plt
from func import *
# Import QFileDialog to let the user choose an image file
from PySide6.QtWidgets import QFileDialog
# Import QTextCodec to change the encoding of sys.stdout
from PySide6 import QtCore
from pathlib import Path
from ui_form import Ui_MainWindow
# Important:
# You need to run the following command to generate the ui_form.py file
#     pyside6-uic form.ui -o ui_form.py, or
#     pyside2-uic form.ui -o ui_form.py

#app = QApplication(sys.argv)

# Reconfigure sys.stdout with a different encoding
sys.stdout.reconfigure(encoding='utf-8', errors='replace')
sys.stdin.reconfigure(encoding='utf-8', errors='replace')

class MainWindow(QMainWindow):
    global img_Mat
#    carplate_haar_cascade = cv2.CascadeClassifier('./haar_cascades/haarcascade_russian_plate_number.xml')
    def __init__(self, parent=None):
        super().__init__(parent)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        self.ui.actionChoose_an_image.triggered.connect(self.open_image_file)
        self.ui.actionChoose_an_image_Tab_2.triggered.connect(self.open_image_file_2)
        #connect the recognitionButton to the function
        self.ui.recognitionButton.clicked.connect(self.recognition)
        self.ui.recognitionButton_2.clicked.connect(self.recognition_2)

    def open_image_file(self):
        global img_Mat
        # Let the user choose an image file to open
        file_info = QFileDialog.getOpenFileUrl(self, 'Open Image', '', 'Images (*.bmp *.jpg *.jpeg *.png)')

        # If the user didn't choose any file, return
        if not file_info or not isinstance(file_info, tuple) or len(file_info) < 1:
            return

        # Extract the local file path from the QUrl
        filepath = file_info[0].toLocalFile()

#        img = cv2.imread(filepath)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        # Convert the image to QImage
#        height, width, channel = img.shape
#        bytesPerLine = 3 * width
#        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
#        # Show the image on the label
#        self.ui.label.setPixmap(QPixmap.fromImage(qImg))

#         Display the image on the label
        display_imagepath_on_label((filepath), self.ui.label)
        img_Mat = cv2.imread(filepath)
        cv2.imshow("img_Mat", img_Mat)
#        img_Mat = qimage_to_cvmat(QImage(filepath))
        display_image_on_label(cvmat_to_qimage(img_Mat), self.ui.label_2)

    def open_image_file_2(self):
        global img_Mat2
        # Let the user choose an image file to open
        file_info = QFileDialog.getOpenFileUrl(self, 'Open Image', '', 'Images (*.bmp *.jpg *.jpeg *.png)')
        # If the user didn't choose any file, return
        if not file_info or not isinstance(file_info, tuple) or len(file_info) < 1:
            return
        # Extract the local file path from the QUrl
        filepath2 = file_info[0].toLocalFile()
#        img = cv2.imread(filepath)
#        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#        # Convert the image to QImage
#        height, width, channel = img.shape
#        bytesPerLine = 3 * width
#        qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
#        # Show the image on the label
#        self.ui.label.setPixmap(QPixmap.fromImage(qImg))
#         Display the image on the label
        display_imagepath_on_label((filepath2), self.ui.label_3)
        img_Mat2 = cv2.imread(filepath2)
#        img_Mat = qimage_to_cvmat(QImage(filepath))
        display_image_on_label(cvmat_to_qimage(img_Mat2), self.ui.label_4)


    def recognition(self):
        global img_Mat
        global license_image
        license_image = cv2.resize(img_Mat, (800, int(800 * img_Mat.shape[0] / img_Mat.shape[1])))

        img1 = preprocess(license_image)
        display_imagepath_on_label("card_img.jpg", self.ui.label_2)
#        display_image_on_label(cvmat_to_qimage(img1), self.ui.label_2)
#        display_Mat_on_label(img1, self.ui.label_2)
#        display_Mat(img1, "After Preprocessing")
        result = recognize_text("card_img.jpg")
        print("The license plate is: " + result)
        #show the result on resultLabel with the text format: "The license plate is: " + result
        self.ui.result_label.setText("The license plate is: " + result)

    def recognition_2(self):
        global img_Mat2
        global license_image2
        detected_image = carplate_detect(img_Mat2)
        #save the image as "carplate_detected.jpg"
        cv2.imwrite("carplate_detected.jpg", detected_image)
        #display the detected image on label_3
        display_imagepath_on_label("carplate_detected.jpg", self.ui.label_3)

        extracted_image = enlarge_img (carplate_extract(img_Mat2), 150)
        #save the image as "carplate_extracted.jpg"
        cv2.imwrite("carplate_extracted.jpg", extracted_image)
        #display the extracted image on label_4
        display_imagepath_on_label("carplate_extracted.jpg", self.ui.label_4)
        
        # Convert image to grayscale
        gray = cv2.cvtColor(extracted_image, cv2.COLOR_BGR2GRAY)
        # Apply median filter
        gray = cv2.medianBlur(gray, 3)
        #Increase contrast
        # gray = cv2.convertScaleAbs(gray, alpha=1.5, beta=0)
        cv2.imshow("gray", gray)
        # Apply adaptive threshold
        # thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 15, 2)
        # cv2.imshow("thresh", thresh)        
        # Display the text extracted from the carplate        
        # print(pytesseract.image_to_string(carplate_extract_img_gray_blur, config =f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))
        self.ui.result_label_2.setText(pytesseract.image_to_string(gray, config =f'--psm 8 --oem 3 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'))

if __name__ == "__main__":
    app = QApplication(sys.argv)
    widget = MainWindow()
    widget.show()
    sys.exit(app.exec())

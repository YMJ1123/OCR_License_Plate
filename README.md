### OCR System for License Plate Recognition

#### Abstract
This readme provides a comprehensive overview of the research project focused on developing an Optical Character Recognition (OCR) system for license plate recognition. The system utilizes Tesseract OCR integrated into a Python program with a PyQt user interface.

#### Introduction
The primary goal of this research is to implement an effective OCR application for license plate recognition using image processing techniques. This section covers the implementation of the OCR algorithm, development of a computer program, and evaluation of the system's accuracy and performance.

#### UI Designs
**UI Layout of Tab 1: OpenCV Color Filtering Method**
- **Description:** This tab showcases the OpenCV color filtering method for license plate recognition.
- **Features:** Image selection, image display, zoom-in functionality, filter process visualization, recognition result display.

**UI Layout of Tab 2: Haar Cascade Method**
- **Description:** This tab demonstrates the use of Haar Cascade for license plate recognition.
- **Features:** Image selection, image display, zoom-in functionality, area highlighting, grayscale processing visualization, recognition result display.

#### Methods
**Simply Use OpenCV Packages and Filtering Methods**
- **Description:** Utilizes OpenCV filters and methods for license plate recognition.
- **Methods:** HSV filter, bitwise_and, GaussianBlur, binary thresholding, erosion, dilation, contourArea, minAreaRect, pytesseract.
- **Limitations:** Limited recognition capability under specific lighting and color conditions.

**Use Haar Cascade as the Pretrained Model**
- **Description:** Implements Haar Cascade for license plate recognition.
- **Methods:** Haar Cascade, similar methods to the previous approach, pytesseract.
- **Limitations:** Effective primarily for Russian license plates.

**Other Attempts**
- **Description:** Attempt to train a YOLO model for Taiwanese license plate recognition.
- **Methods:** YOLO training.
- **Limitations:** Poor performance due to limited training time.

#### Program Links
Executable Link: [OCR System](https://ntucc365-my.sharepoint.com/:u:/g/personal/b09611009_ntu_edu_tw/Ea1dZc2pWrZDghCKHP162T4Bv9InL9Vgb4VK1EjT_pezRg?e=TBsajV)
Test Image: China3.jpg for Tab 1, Russia2.jpg for Tab 2

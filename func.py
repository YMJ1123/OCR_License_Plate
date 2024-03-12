import numpy as np
import cv2
from matplotlib import pyplot as plt
#import QImage, QPixmap
from PySide6.QtWidgets import QApplication, QMainWindow, QFileDialog, QLabel
from PySide6.QtGui import QImage, QPixmap, qRgb
from PySide6.QtCore import Qt
from PySide6.QtGui import QImageReader
from PySide6.QtCore import QRectF
#import Qdebug
from PySide6.QtCore import qDebug
import pytesseract # This is theTesseractOCR Python

# Set Tesseract CMD path to the location oftesseract.exe file
# pytesseract.pytesseract.tesseract_cmd =r'.\OCR\tesseract.exe'

# Show the image on a assigned label
#def show_image(img, label):
#    # Convert the image to RGB format
#    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
#    # Convert the image to QImage
#    height, width, channel = img.shape
#    bytesPerLine = 3 * width
#    qImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)
#    # Show the image on the label
#    label.setPixmap(QPixmap.fromImage(qImg))

def display_imagepath_on_label(image, label):
    label_size = label.size()
#    # Set the pixmap with the label size while keeping the aspect ratio
#    label.setPixmap(QPixmap.fromImage(image.scaled(label_size, Qt.KeepAspectRatio)))
# Use QImageReader to get the original image size and pixel density
    reader = QImageReader(image)
    reader.setAutoTransform(True)  # Automatically transform based on Exif orientation
    # Get the original image size
    original_size = reader.size()

    # Calculate the scaled size to fit the label while keeping the aspect ratio
    scaled_size = QPixmap(original_size).scaled(label_size, Qt.KeepAspectRatio).size()

    # Calculate the position to center the image in the label
    position = QRectF(label_size.width() - scaled_size.width(), label_size.height() - scaled_size.height(), scaled_size.width(), scaled_size.height())

    # Set the pixmap with the label size while keeping the aspect ratio
    pixmap = QPixmap.fromImageReader(reader)
    pixmap.setDevicePixelRatio(QImage(image).devicePixelRatio())
    pixmap = pixmap.scaled(scaled_size, Qt.KeepAspectRatio, Qt.SmoothTransformation)

    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)

def display_image_on_label(image, label):
    # Get the size of the label
    label_size = label.size()

    # Set the pixmap with the label size while keeping the aspect ratio
    label.setPixmap(QPixmap.fromImage(image.scaled(label_size, Qt.KeepAspectRatio)))

def qimage_to_cvmat(in_image, in_clone_image_data=True):
    image_format = in_image.format()

    if image_format in [QImage.Format_ARGB32, QImage.Format_ARGB32_Premultiplied]:
        mat = np.array(in_image.bits()).reshape(in_image.height(), in_image.width(), 4)
        return mat if in_clone_image_data else mat.copy()

    elif image_format in [QImage.Format_RGB32, QImage.Format_RGB888]:
        if not in_clone_image_data:
            print("Warning: Conversion requires cloning because we use a temporary QImage")

        swapped = in_image.convertToFormat(QImage.Format_RGB888)
        swapped = swapped.rgbSwapped()

        mat = np.array(swapped.bits()).reshape(swapped.height(), swapped.width(), 3)
        return mat.copy()

    elif image_format == QImage.Format_Indexed8:
        mat = np.array(in_image.bits()).reshape(in_image.height(), in_image.width())
        return mat if in_clone_image_data else mat.copy()

    elif image_format == QImage.Format_Grayscale8:
        mat = np.array(in_image.bits()).reshape(in_image.height(), in_image.width())
        return mat if in_clone_image_data else mat.copy()

    else:
        print("Warning: QImage format not handled:", image_format)

    return None
#def cvmat_to_qimage(data):
#    # 8-bits unsigned, NO. OF CHANNELS=1
#    if data.dtype == np.uint8:
#        channels = 1 if len(data.shape) == 2 else data.shape[2]
#    if channels == 3: # CV_8UC3
#        # Copy input Mat
#        # Create QImage with same dimensions as input Mat
#        img = QImage(data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_RGB888)
#        return img.rgbSwapped()
#    elif channels == 1:
#        # Copy input Mat
#        # Create QImage with same dimensions as input Mat
#        img = QImage(data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_Indexed8)
#        return img
#    else:
#        qDebug("ERROR: numpy.ndarray could not be converted to QImage. Channels = %d" % data.shape[2])
#        return QImage()


def cvmat_to_qimage(data):
    # 8-bits unsigned, NO. OF CHANNELS=1
    if data.dtype == np.uint8:
        channels = 1 if len(data.shape) == 2 else data.shape[2]
    else:
        channels = 0  # 如果不是 uint8 类型，将 channels 设为 0

    if channels == 3:  # CV_8UC3
        # Copy input Mat
        # Create QImage with same dimensions as input Mat
        img = QImage(data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_RGB888)
        return img.rgbSwapped()
    elif channels == 1:
        # Copy input Mat
        # Create QImage with same dimensions as input Mat
        img = QImage(data, data.shape[1], data.shape[0], data.strides[0], QImage.Format_Indexed8)
        return img
    else:
        qDebug("ERROR: numpy.ndarray could not be converted to QImage. Channels = %d" % channels)
        return QImage()

def display_Mat_on_label(mat, label):
    # Convert Mat to QImage
    if mat.dtype == np.uint8:
        channels = 1 if len(mat.shape) == 2 else mat.shape[2]
    else:
        channels = 0  # If not uint8, set channels to 0

    if channels == 3:  # CV_8UC3
        qimage = QImage(mat.data, mat.shape[1], mat.shape[0], mat.strides[0], QImage.Format_RGB888)
        qimage = qimage.rgbSwapped()
    elif channels == 1:
        qimage = QImage(mat.data, mat.shape[1], mat.shape[0], mat.strides[0], QImage.Format_Indexed8)
    else:
        qDebug("ERROR: Mat could not be converted to QImage. Channels = %d" % channels)
        return

    # Convert QImage to QPixmap and set on QLabel
    pixmap = QPixmap.fromImage(qimage)
    label.setPixmap(pixmap)
    label.setAlignment(Qt.AlignCenter)

def display_Mat(mat, window_title):
    #Make the window resizable
    cv2.namedWindow(window_title, cv2.WINDOW_NORMAL)
    cv2.imshow(window_title, mat)
    cv2.waitKey(0)  # Wait for a key press before closing the window
    cv2.destroyAllWindows()

def preprocess(license_image): #preprocessing image for license plate recognition
#    license_image = cv2.resize(src, (800, int(800 * src.shape[0] / src.shape[1])))
#    display_image_on_label(license, ui.label_2)
    license_prepared = license_prepation(license_image)
    contours, hierarchy = cv2.findContours(license_prepared, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    license_area = choose_license_area(contours, 1000)
    result = license_segment(license_area, license_image)
    return result

def license_prepation(image):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 从RGB图像转为hsv色彩空间
#    low_hsv = np.array([49, 23, 0])  # 设置颜色
#    high_hsv = np.array([127, 49, 37])
    low_hsv = np.array([108, 43, 46])  # 设置颜色:順序為H、S、V CHINA
    high_hsv = np.array([124, 255, 255])
    #HK
#    low_hsv = np.array([22, 145, 121])  # 设置颜色:順序為H、S、V
#    high_hsv = np.array([72, 255, 255])
    #設定白色的HSV範圍
#    low_hsv = np.array([87, 42, 131])  # 设置颜色:順序為H、S、V
#    high_hsv = np.array([107, 82, 193])
    mask = cv2.inRange(image_hsv, lowerb=low_hsv, upperb=high_hsv)  # 选出蓝色的区域
#    mask = cv2.inRange(image_hsv, (49, 23, 0), (127, 49, 37))
    cv2.imshow('mask', mask)
    image_dst = cv2.bitwise_and(image, image, mask=mask)  # 取frame与mask中不为0的相与，在原图中扣出蓝色的区域，mask=mask必须有
    cv2.imshow('license_dst', image_dst)
    image_blur = cv2.GaussianBlur(image_dst, (7, 7), 0)#高斯模糊，消除噪声。第二个参数为卷积核大小，越大模糊的越厉害
    cv2.imshow('license_blur',image_blur)
    image_gray = cv2.cvtColor(image_blur, cv2.COLOR_BGR2GRAY)#转为灰度图像
    ret, binary = cv2.threshold(image_gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)#二值化
    cv2.imshow('binary', binary)
    # 以下为形态学操作
    # 闭操作，先膨胀再腐蚀
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (4, 6))#得到一个4*6的卷积核
    image_opened = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel1)#开操作，去一些干扰
    cv2.imshow('license_opened', image_opened)
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))#得到一个7*7的卷积核
    image_closed = cv2.morphologyEx(image_opened, cv2.MORPH_CLOSE, kernel2)#闭操作，填充一些区域
    cv2.imshow('license_closed', image_closed)
    return image_closed

def choose_license_area(contours, Min_Area):
    temp_contours = []
    for contour in contours:
        if cv2.contourArea(contour) > Min_Area:  # 面积大于MIN_AREA的区域保留
            temp_contours.append(contour)
    license_area = []
    for temp_contour in temp_contours:
        rect_tupple = cv2.minAreaRect(temp_contour)
        # print(rect_tupple)
        rect_width, rect_height = rect_tupple[1]  # 0为中心点，1为长和宽，2为角度
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 车牌正常情况下宽高比在2 - 5.5之间
        if aspect_ratio > 2 and aspect_ratio < 5.5:
            license_area.append(temp_contour)
    return license_area

def license_segment(license_area, license_image):

    print("Area num:", len(license_area))
    n = 1
    for a in license_area:
        print("Area no.", n, ":", cv2.contourArea(a))
#    if (len(license_area)) == 1:
#        for car_plate in license_area:
#            row_min, col_min = np.min(car_plate[:, 0, :], axis=0)  # 行是row 列是col
#            row_max, col_max = np.max(car_plate[:, 0, :], axis=0)#这两行代码为了找出车牌位置的坐标
#            card_img = license_image[col_min:col_max, row_min:row_max, :]
#            # cv2.imshow("card_img", card_img)
#            cv2.imwrite("card_img.jpg", card_img)
#    else:
#        # 如果不止一個區域，則遍歷每一個區域，找出最大的區域
#        max_area = 0
#        max_num = 0
#        x = 1
#        for i in license_area:
#            if i.shape[0] * i.shape[1] > max_area:
#                max_area = i.shape[0] * i.shape[1]
#                max_num = x
#            x += 1

#        # 將最大的區域保存為car_plate
#        car_plate = license_area[max_num - 1]
#        # 找出車牌的位置並保存
#        row_min, col_min = np.min(car_plate[:, 0, :], axis=0)  # 行是row 列是col
#        row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
#        card_img = license_image[col_min:col_max, row_min:row_max, :]
#        # 如果不止一個區域，則選取最接近矩形的區域
#        # 遍歷每一個區域，找出最接近矩形的區域
    max_area = 0
    max_num = 0
    x = 1
    for i in license_area:
        # 找出該區域的最小外接矩形
        rect_tupple = cv2.minAreaRect(i)
        # print(rect_tupple)
        rect_width, rect_height = rect_tupple[1]
        if rect_width < rect_height:
            rect_width, rect_height = rect_height, rect_width
        aspect_ratio = rect_width / rect_height
        # 車牌正常情況下宽高比在2 - 5.5之間
        if aspect_ratio > 2 and aspect_ratio < 5.5:
            if i.shape[0] * i.shape[1] > max_area:
                max_area = i.shape[0] * i.shape[1]
                max_num = x
        x += 1

    # 將最大的區域保存為car_plate
    car_plate = license_area[max_num - 1]
    # 找出車牌的位置並保存
    row_min, col_min = np.min(car_plate[:, 0, :], axis=0)
    row_max, col_max = np.max(car_plate[:, 0, :], axis=0)
    #取稍微大一點的區域
    row_min = row_min - 5
    row_max = row_max + 5
    col_min = col_min - 5
    col_max = col_max + 5

    card_img = license_image[col_min:col_max, row_min:row_max, :]
    #把形變的矩形還原
    card_img = cv2.resize(card_img, (int(card_img.shape[1] * 2.5), int(card_img.shape[0] * 2.5)))
    cv2.imwrite("card_img.jpg", card_img)
    return card_img

def recognize_text(imagepath):
    image = cv2.imread(imagepath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)#转为灰度图片
    ret, binary = cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY_INV)#二值化
    cv2.imshow('bin', binary)  #显示二值过后的结果， 白底黑字
    bin1 = cv2.resize(binary, (370, 82))#改变一下大小，有助于识别
    kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 5))#获取一个卷积核，参数都是自己调的
    dilated = cv2.dilate(bin1, kernel1)  # 白色区域膨胀
    #使用pytesseract進行識別(支援簡體中文)
    text = pytesseract.image_to_string(dilated, lang='chi_sim')
    return text
#############################以下為Tab2的函數###############################################
#import the XML file for the car plate detection
cascade_src = 'haarcascade_license_plate_rus_16stages.xml'
carplate_haar_cascade = cv2.CascadeClassifier(cascade_src)

def carplate_detect(image):
    carplate_overlay = image.copy()
    carplate_rects = carplate_haar_cascade.detectMultiScale(carplate_overlay,scaleFactor=1.1,minNeighbors=3)
    for (x,y,w,h) in carplate_rects:
        cv2.rectangle(carplate_overlay, (x,y), (x+w,y+h), (0,255,0), 5)
    return carplate_overlay

# def carplate_extract(image):
#     carplate_rects = carplate_haar_cascade.detectMultiScale(image,scaleFactor=1.1,minNeighbors=5)

#     for (x,y,w,h) in carplate_rects:
#         carplate_img = image[y+15:y+h-10 ,x+15:x+w-20] # Adjusted to extractspecific region of interest i.e. car license plate

#     return carplate_img

def carplate_extract(image):
    carplate_rects = carplate_haar_cascade.detectMultiScale(image, scaleFactor=1.1, minNeighbors=3)

    # Find the largest rectangle
    max_area = 0
    max_rect = None
    for (x, y, w, h) in carplate_rects:
        area = w * h
        if area > max_area:
            max_area = area
            max_rect = (x, y, w, h)

    if max_rect is not None:
        (x, y, w, h) = max_rect
        carplate_img = image[y + 15:y + h - 10, x + 15:x + w - 20]  # Adjusted to extract specific region of interest i.e. car license plate

        return carplate_img
    else:
        return None


# Enlarge image for further processinglater on
def enlarge_img(image, scale_percent):
    width = int(image.shape[1] * scale_percent / 100)
    height = int(image.shape[0] * scale_percent / 100)
    dim = (width,height)
    resized_img = cv2.resize(image, dim, interpolation = cv2.INTER_AREA)
    return resized_img




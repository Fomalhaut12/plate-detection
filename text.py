import pytesseract
import imutils
import cv2
import os
import numpy as np
import easyocr
import skimage.measure as measure
from svm_train import *
pytesseract.pytesseract.tesseract_cmd = r'D:\Program Files\Tesseract-OCR\tesseract' #CHANGE THIS BEFORE COMMIT

#导入SVM
svm_model = SVM(C=1, gamma=0.5)
model_1,model_2 = svm_model.train_svm()

#使用OCR
def ocr():
    image_dir = './plates'
    num = 0
    for root, dirs, files in os.walk(image_dir):
        for filename in files:
            if filename.endswith('.jpg') or filename.endswith('.png'):
                image_path = os.path.join(root, filename)
                img = cv2.imread(image_path)
                num += 1
                img = imutils.resize(img, width=400, height=400)
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                gray = cv2.GaussianBlur(gray, (3, 3), 0)
                plate = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
                cv2.imshow('plate',plate)
                cv2.waitKey(0)
                reader = easyocr.Reader(['ch_sim', 'en'])
                #text = pytesseract.image_to_string(plate, lang='chi_sim+eng')
                text = reader.readtext(plate)
                print(f"{num}:",text)

#字符分割法
def in_bboxes(bbox, bboxes):
    for bb in bboxes:
        minr0, minc0, maxr0, maxc0 = bb
        minr1, minc1, maxr1, maxc1 = bbox
        if minr1 >= minr0 and maxr1 <= maxr0 and minc1 >= minc0 and maxc1 <= maxc0:
            return True
    return False

def get_chars(regions,morph_img,thresh):
    bboxes = []
    for props in regions:
        y0, x0 = props.centroid  # 获取连通区域的质心
        minr, minc, maxr, maxc = props.bbox # 获取连通区域边界框
        # 如果边界框宽度超过图像宽度的1/5或者高度小于图像高度的1/3，则忽略该区域
        if maxc - minc > morph_img.shape[1] / 5 or maxr - minr < morph_img.shape[0] / 3:
            continue

        # 如果该区域已经被包含在之前的边界框中，则忽略该区域
        bbox = [minr, minc, maxr, maxc]
        if in_bboxes(bbox, bboxes):
            continue
        # 如果该区域的质心与图像中心的纵向距离超过图像高度的1/3，则忽略该区域
        if abs(y0 - morph_img.shape[0] / 2) > morph_img.shape[0] / 3:
            continue

        bboxes.append(bbox)

    #从左到右
    bboxes = sorted(bboxes, key=lambda x: x[1])

    chars = []
    for i, bbox in enumerate(bboxes):
        minr, minc, maxr, maxc = bbox
        ch = thresh[minr:maxr, minc:maxc]
        chars.append(ch)
        #cv2.imshow('ch',ch)
        #cv2.waitKey(0)

    return chars

def isgreen(img): # 判断是否为绿色车牌
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    lower_green = (36, 25, 25)
    upper_green = (86, 255, 255)
    mask = cv2.inRange(hsv, lower_green, upper_green)
    green_pixels = cv2.countNonZero(mask)
    total_pixels = img.shape[0] * img.shape[1]
    green_ratio = green_pixels / total_pixels
    return green_ratio > 0.2

def readtext(img): # 输入车牌图片，输出识别文本。为main的接口
    img = imutils.resize(img, width=400)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if isgreen(img):
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    else:
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    canny = cv2.Canny(thresh, 350, 400)
    kernel = np.ones((3, 3), np.uint8)
    morph_img = cv2.dilate(canny, kernel, iterations=2)
    # cv2.imshow('m',morph_img)
    # cv2.waitKey(0)

    # 连通区域标记
    label_img = measure.label(morph_img)
    regions = measure.regionprops(label_img)

    pre = []
    chars = get_chars(regions,morph_img,thresh)
    for i, part_card in enumerate(chars):

        # 去掉不是字符的边缘
        ratio = part_card.shape[1] / part_card.shape[0]
        if np.mean(part_card) < 255 / 5 or ratio < 0.15:
            continue

        w = abs(part_card.shape[1] - SZ) // 2
        part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])  # 用来给图片添加边框
        # 处理图像，便于SVM判断
        part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
        part_card = deskew(part_card)
        part_card = preprocess_hog([part_card])
        # 第一个字符为中文，其余为英文
        if i == 0:
            resp = model_2.predict(part_card)
            character = provinces[int(resp[0]) - PROVINCE_START]
        else:
            resp = model_1.predict(part_card)
            character = chr(int(resp[0]))

        pre.append(character)
    return ''.join(pre)


# 测试用
'''
image_dir = './plates'


for root, dirs, files in os.walk(image_dir):
    for filename in files:
        path = os.path.join(root, filename)
        img = cv2.imread(path)
        #图片预处理
        img = imutils.resize(img, width=400)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        if filename == '1-2.jpg' or filename == '3-2.jpg':
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        else:
            thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        canny = cv2.Canny(thresh, 350, 400)
        kernel = np.ones((3, 3), np.uint8)
        morph_img = cv2.dilate(canny, kernel, iterations=2)
        #cv2.imshow('m',morph_img)
        #cv2.waitKey(0)


        # 连通区域标记
        label_img = measure.label(morph_img)
        regions = measure.regionprops(label_img)

        pre = []
        chars = get_chars(regions,morph_img,thresh)
        for i, part_card in enumerate(chars):

            #去掉不是字符的边缘
            ratio = part_card.shape[1]/part_card.shape[0]
            if np.mean(part_card) < 255 / 5 or ratio < 0.15:
                continue

            w = abs(part_card.shape[1] - SZ) // 2
            part_card = cv2.copyMakeBorder(part_card, 0, 0, w, w, cv2.BORDER_CONSTANT, value=[0, 0, 0])  #用来给图片添加边框
            #处理图像，便于SVM判断
            part_card = cv2.resize(part_card, (SZ, SZ), interpolation=cv2.INTER_AREA)
            part_card = deskew(part_card)
            part_card = preprocess_hog([part_card])
            #第一个字符为中文，其余为英文
            if i == 0:
                resp = model_2.predict(part_card)
                character = provinces[int(resp[0]) - PROVINCE_START]
            else:
                resp = model_1.predict(part_card)
                character = chr(int(resp[0]))

            pre.append(character)
        print(f'{filename}:',''.join(pre)) '''


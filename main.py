from plate_detect import ANPR
from imutils import paths
import pytesseract
import imutils
import cv2
import os
from PIL import Image, ImageDraw, ImageFont
from text import *

def save_img(name, image):
    try:  # in a folder and also the image of the extracted ROI.
        image_dir = 'res'
        os.chdir(image_dir)
    except:
        print("No folder directory found. Creating...")
        os.mkdir(image_dir)
        os.chdir(image_dir)

    cv2.imwrite(name, image)
    os.chdir('..')

input_dir = './images/medium'
input_dir_name = input_dir.split('/')[-1]
anpr = ANPR(input_dir_name, maxratio=4.5, minratio=2, maxarea=10000, minarea=1500, debug=False)
num = 0
imagePaths = sorted(list(paths.list_images(input_dir)))

for imagePath in imagePaths:
    num += 1
    image = cv2.imread(imagePath)
    if input_dir == './images/easy':
        lpText = readtext(image)
        print(f"plate number: {lpText}")
        continue

    image = imutils.resize(image, width=400, height=400)
    (plate, lpCnt) = anpr.find(image) # 定位车牌

    if plate is not None and lpCnt is not None:
        lpText = readtext(plate)  # 识别文本
        # lpText = pytesseract.image_to_string(plate,lang='chi_sim+eng')
        save_img(f"plate{num}.jpg", plate)  # 保存车牌

        box = cv2.boxPoints(cv2.minAreaRect(lpCnt))  # 框选车牌
        box = box.astype("int")
        cv2.drawContours(image, [box], -1, (0, 255, 0), 2)
        (x, y, w, h) = cv2.boundingRect(lpCnt)

        #使用PIL库来在图片上绘制文本
        pil_img = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        font = ImageFont.truetype('simsun.ttc', size=25)
        draw = ImageDraw.Draw(pil_img)
        draw.text((x, y-25), lpText, font=font, fill=(0, 255, 0))
        image = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        cv2.imshow('img',image)
        cv2.waitKey(0)

        save_img(f"Final{num}.jpg", image)
        print(f"plate number: {lpText}")

    cv2.destroyAllWindows()


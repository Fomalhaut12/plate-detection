import imutils
import cv2


class ANPR:
    def __init__(self, input_dir, minratio=2, maxratio=4.5, minarea=3000, maxarea=10000, debug=False):
        self.minratio = minratio   #长宽比上下限
        self.maxratio = maxratio
        self.minarea = minarea    #面积上下限
        self.maxarea = maxarea
        self.debug = debug     #是否展示图像
        self.input_dir = input_dir


    def debug_imshow(self, title, image,waitKey=False):
        if self.debug:  # 用于展示图像
            cv2.imshow(title, image)
            #cv2.imwrite(f'{title}.jpg', image)
            if waitKey:
                cv2.waitKey(0)


    def morphology(self, gray, rectKern):

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rectKern)

        squareKern = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, squareKern)

        light = cv2.threshold(light, 0, 255,
                              cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow('light',light,waitKey=True)
        return [blackhat, light]



    def possible_license_plates(self, gray, image, keep=5):
        rectKern = cv2.getStructuringElement(cv2.MORPH_RECT, (13, 5))
        morphology = self.morphology(gray, rectKern)
        morph = morphology[0]

        canny = cv2.Canny(morph, 350, 450)  # 边缘检测

        gaussian = cv2.GaussianBlur(canny, (5, 5), 0)
        gaussian = cv2.morphologyEx(gaussian, cv2.MORPH_CLOSE, rectKern)
        thresh = cv2.threshold(gaussian, 0, 255,
                               cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        self.debug_imshow("Grad Thresh", thresh, waitKey=True)

        cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cnts = imutils.grab_contours(cnts)
        cnts = sorted(cnts, key=cv2.contourArea, reverse=True)[:keep]

        img = image.copy()
        for c in cnts:
            cv2.drawContours(img, [c], -1, 255, 2)
            self.debug_imshow("Contours", img)

        #self.debug_imshow("Masked", thresh, waitKey=True)

        return cnts

    def locate_license_plate(self, image, gray, candidates):
        lpCnt = None
        pl = None

        for c in candidates:
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = w / float(h)
            area = w * h
            #print(ratio, area)
            if self.minratio <= ratio <= self.maxratio and self.minarea <= area <= self.maxarea:
                lpCnt = c
                licensePlate = gray[y:y + h, x:x + w]
                pl = image[y:y + h, x:x + w]
                self.debug_imshow('plate',pl,waitKey=True)

                break
        return (pl, lpCnt)



    def find(self, image):  # 输入图像，输出检测车牌，为main的接口

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        candidates = self.possible_license_plates(gray, image, 5)
        (lp, lpCnt) = self.locate_license_plate(image, gray, candidates)

        if lp is not None:
            self.debug_imshow("License Plate", lp, waitKey=True)
            return (lp, lpCnt)

        return None,None

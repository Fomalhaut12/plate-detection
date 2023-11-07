import cv2
import numpy as np
from numpy.linalg import norm
import os


SZ = SZ = 20 
PROVINCE_START = 1000

provinces = [
"zh_cuan", "川",
"zh_e", "鄂",
"zh_gan", "赣",
"zh_gan1", "甘",
"zh_gui", "贵",
"zh_gui1", "桂",
"zh_hei", "黑",
"zh_hu", "沪",
"zh_ji", "冀",
"zh_jin", "津",
"zh_jing", "京",
"zh_jl", "吉",
"zh_liao", "辽",
"zh_lu", "鲁",
"zh_meng", "蒙",
"zh_min", "闽",
"zh_ning", "宁",
"zh_qing", "靑",
"zh_qiong", "琼",
"zh_shan", "陕",
"zh_su", "苏",
"zh_sx", "晋",
"zh_wan", "皖",
"zh_xiang", "湘",
"zh_xin", "新",
"zh_yu", "豫",
"zh_yu1", "渝",
"zh_yue", "粤",
"zh_yun", "云",
"zh_zang", "藏",
"zh_zhe", "浙"
]


def deskew(img):
	# 计算图像的矩
	m = cv2.moments(img)
	# 如果mu02值的绝对值小于0.01，则返回图像的副本
	if abs(m['mu02']) < 1e-2:
		return img.copy()
	# 计算图像的倾斜角度
	skew = m['mu11']/m['mu02']
	# 构建2x3变换矩阵
	M = np.float32([[1, skew, -0.5*SZ*skew], [0, 1, 0]])
	# 对图像进行仿射变换
	img = cv2.warpAffine(img, M, (SZ, SZ), flags=cv2.WARP_INVERSE_MAP | cv2.INTER_LINEAR)
	return img


def preprocess_hog(digits):
	samples = []
	for img in digits:
		# 使用Sobel算子计算图像的梯度
		gx = cv2.Sobel(img, cv2.CV_32F, 1, 0)
		gy = cv2.Sobel(img, cv2.CV_32F, 0, 1)
		# 计算梯度的幅值和方向
		mag, ang = cv2.cartToPolar(gx, gy)
		# 将方向转换成0~15的整数，将图像划分成4个小的cell，每个cell的大小为10x10
		bin_n = 16
		bin = np.int32(bin_n*ang/(2*np.pi))
		bin_cells = bin[:10,:10], bin[10:,:10], bin[:10,10:], bin[10:,10:]
		mag_cells = mag[:10,:10], mag[10:,:10], mag[:10,10:], mag[10:,10:]
		# 对于每个cell，计算它的梯度方向直方图
		hists = [np.bincount(b.ravel(), m.ravel(), bin_n) for b, m in zip(bin_cells, mag_cells)]
		# 把所有cell的直方图拼接在一起，得到整个图像的HOG特征向量
		hist = np.hstack(hists)
		
		# 将特征向量进行Hellinger核变换，使得特征更加鲁棒
		eps = 1e-7
		hist /= hist.sum() + eps
		hist = np.sqrt(hist)
		hist /= norm(hist) + eps
		
		samples.append(hist)
	return np.float32(samples)


class StatModel(object):
	def load(self, fn):
		self.model = self.model.load(fn)  
	def save(self, fn):
		self.model.save(fn)

class SVM(StatModel):
	def __init__(self, C = 1, gamma = 0.5):
		self.model = cv2.ml.SVM_create()
		self.model.setGamma(gamma)
		self.model.setC(C)
		self.model.setKernel(cv2.ml.SVM_RBF)
		self.model.setType(cv2.ml.SVM_C_SVC)
	# 训练svm
	def train(self, samples, responses):
		self.model.train(samples, cv2.ml.ROW_SAMPLE, responses)
	# 预测
	def predict(self, samples):
		r = self.model.predict(samples)
		return r[1].ravel()

	def train_svm(self):
		#识别英文字母和数字
		self.model = SVM(C=1, gamma=0.5)
		#识别中文
		self.modelchinese = SVM(C=1, gamma=0.5)
		if os.path.exists("./train_dat/svm.dat"):
			self.model.load("./train_dat/svm.dat")
		else:
			chars_train = []
			chars_label = []
			
			for root, dirs, files in os.walk("./train/chars"):
				if len(os.path.basename(root)) > 1:
					continue
				root_int = ord(os.path.basename(root))
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					#chars_label.append(1)
					chars_label.append(root_int)
			
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)

			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.model.train(chars_train, chars_label)

		if os.path.exists("./train_dat/svmchinese.dat"):
			self.modelchinese.load("./train_dat/svmchinese.dat")
		else:
			chars_train = []
			chars_label = []
			for root, dirs, files in os.walk("./train/charsChinese"):
				if not os.path.basename(root).startswith("zh_"):
					continue
				pinyin = os.path.basename(root)
				index = provinces.index(pinyin) + PROVINCE_START + 1
				for filename in files:
					filepath = os.path.join(root,filename)
					digit_img = cv2.imread(filepath)
					digit_img = cv2.cvtColor(digit_img, cv2.COLOR_BGR2GRAY)
					chars_train.append(digit_img)
					chars_label.append(index)
			chars_train = list(map(deskew, chars_train))
			chars_train = preprocess_hog(chars_train)

			chars_label = np.array(chars_label)
			print(chars_train.shape)
			self.modelchinese.train(chars_train, chars_label)

		return self.model, self.modelchinese


if __name__ == "__main__":
	svm_model = SVM(C=1, gamma=0.5)

	model_1,model_2 = svm_model.train_svm()
	print(model_1)
	print(model_2)
	if not os.path.exists('./train_dat'):
		os.makedirs('./train_dat')
	model_1.save("./train_dat/svm.dat")
	model_2.save("./train_dat/svmchinese.dat")
	
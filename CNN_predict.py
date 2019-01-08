import numpy as np
import cv2
import skimage.io
import matplotlib.pyplot as plt
from keras.models import load_model

number = ['BLACK_CHE','BLACK_JIANG']
model=load_model('./model.h5')
model.load_weights('./weights.h5')
image_path ='./data/train/BLACK_JIANG/BLACK_JIANG_0_6.jpg'
# 加载图像
img=cv2.imread(image_path,cv2.IMREAD_COLOR)
img2=cv2.resize(img,(150,150),interpolation=cv2.INTER_CUBIC)
# print(img2.shape)
# cv2.namedWindow('img',cv2.WINDOW_AUTOSIZE)
# cv2.imshow('img',img2)
# k=cv2.waitKey(0)
# if k==27:
#     cv2.destroyAllWindows()
#     plt.close()

img2 = np.expand_dims(img2, axis=0)
# print(img2.shape)

# # 对数字进行预测
predict = model.predict(img2, verbose=0)
index = np.argmax(predict)
print(predict)
print('物体为：%s'%number[index])
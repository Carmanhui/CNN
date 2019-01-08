
# coding: utf-8

# In[ ]:


import math
import os
from PIL import Image
 
# 功能：博客图片生成缩略图，1280横屏压缩，1000竖屏压缩
# 参数：图片名称
# 返回：OK，保存同名文件在thumb目录下
def JfzBlogImgThumb(ImgName):
    im = Image.open(ImgName)
    print('格式',im.format, '，分辨率',im.size, '，色彩',im.mode)
    if max(im.size[0], im.size[1]) > 1000:
        if im.size[0] > im.size[1]:
            im.thumbnail((1280, 1280))
        else:
            im.thumbnail((1000, 1000))
        os.mkdir('thumb')
        im.save('thumb\\'+ImgName, 'JPEG', quality=90)
    return 'OK'
 
# JfzBlogImgThumb('1.jpg')
# JfzBlogImgThumb('2.jpg')
 
# 列出当前目录所有jpg文件
lst=os.listdir(os.getcwd())
imgname = [c for c in lst if os.path.isfile(c) and c.endswith('.jpg')]
print(imgname)
 
# 全处理
OutCheck = map(JfzBlogImgThumb,imgname)
print(list(OutCheck))


import os
from PIL import Image
import platform
import random
import cv2
import numpy as np


train_annotation = '../2007_train.txt'


symbol = '/'
plat = platform.system().lower()
if plat == 'windows':
    symbol='\\'

def getcount(path):
    count = 0
    for root, dirs, files in os.walk(path):
        for each in files:
            count += 1
    return count

VOCdevkit_path = 'VOCdevkit'
segtarget = 'Temptarget'
padding = 10
def preparation(train_lines):

    if not os.path.exists(VOCdevkit_path+symbol+segtarget):
        os.makedirs(VOCdevkit_path+symbol+segtarget)

    countfile = getcount(VOCdevkit_path+symbol+segtarget)
    if countfile <1:
        countSEG= 0
        for i in train_lines:
            info = i.split(' ')

            if len(info) < 2:
                continue
            image   = Image.open(info[0])
            print('正在处理 '+ info[0] + '中的数据........')
            boxs = info[1:]
            for ii in boxs:
                xy = ii.split(',')
                xyint = [int(xy[0])-padding,int(xy[1])-padding,int(xy[2])+padding,int(xy[3])+padding]
                segimage = image.crop((xyint[0] , xyint[1], xyint[2],xyint[3]))
                name = str(int(xy[4]))
                imagepath = VOCdevkit_path + symbol + segtarget+symbol+name
                if not os.path.exists(imagepath):
                    os.makedirs(imagepath)
                imagepath = imagepath+symbol+str(countSEG)+info[0][-4:]
                countSEG =countSEG+1
                segimage.save(imagepath)
    else:
        print('此操作已经执行过，且'+VOCdevkit_path+symbol+segtarget+' 路径下存在文件，请自行确认此次数据截取时候的 padding 和生成截取图像时候的 padding 值相同')

def traindatapreparation(train_annotation_path):
    with open(train_annotation_path, encoding='utf-8') as f:
        train_lines = f.readlines()
    preparation(train_lines)



def Picture_Synthesis(M_Img, S_Img, coordinate=None):
    M_Img.paste(S_Img, coordinate, mask=None)
    return M_Img




def bb_overlab(x1, y1, w1, h1, x2, y2, w2, h2):
    if(x1>x2+w2):
        return 0
    if(y1>y2+h2):
        return 0
    if(x1+w1<x2):
        return 0
    if(y1+h1<y2):
        return 0
    return 1

tinydict={}
upperLimit = 4
def getimage(annotation_line,num=5):

    line = annotation_line.split()
    image = Image.open(line[0])
    allboxs = []
    names = []
    for box in line[1:]:
        boxs = box.split(',')
        allboxs.append(boxs)
        name = boxs[4]

        path = line[0][:line[0].rfind(VOCdevkit_path)]
        imagepath = path +VOCdevkit_path + symbol + segtarget + symbol + name
        if not name in tinydict:
            numfile = getcount(imagepath)
            tinydict[name] = numfile
        if not name in names:
            names.append(name)

    for name in names:
        random_nums_list = []
        while len(random_nums_list) < num:
            randomnum = str(random.randint(1, tinydict[name]-1))
            if not randomnum in random_nums_list:
                random_nums_list.append(randomnum)

        for i in range(len(random_nums_list)):
            eachimagepath = imagepath + symbol +random_nums_list[i]+line[0][-4:]
            eachimage =  Image.open(eachimagepath)
            M_Img_w, M_Img_h = image.size
            S_Img_w, S_Img_h = eachimage.size

            flag = True
            count = 0
            while flag :
                newx = random.randint(20, M_Img_w-S_Img_w-20)
                newy = random.randint(20, M_Img_h-S_Img_h-20)
                isok = True
                for ii in range(len(allboxs)):
                    x_0, y_0, x_1, y_1 = int(allboxs[ii][0]), int(allboxs[ii][1]), int(allboxs[ii][2]), int(allboxs[ii][3])
                    result = bb_overlab(newx, newy, S_Img_w, S_Img_h, x_0, y_0, (x_1-x_0),(y_1-y_0))
                    if result >0.0:
                        isok = False
                        break;
                if isok:
                    image = Picture_Synthesis(image, eachimage, (newx, newy))
                    # image = image.paste( eachimage, (newx, newy), mask=None)

                    allboxs.append([str(newx+padding), str(newy+padding), str(newx+S_Img_w-padding), str(newy+S_Img_h-padding), str(name)])
                    count = 0
                    flag = False
                else:
                    count = count +1
                    if count == upperLimit :
                        flag = False

    return image, allboxs

def verify(path, allboxs):
    im = cv2.imread(path)
    savepath =path[:-4]+'_01'+path[-4:]
    for i in range(len(allboxs)):
        minX = int(allboxs[i][0])
        minY = int(allboxs[i][1])
        maxX = int(allboxs[i][2])
        maxY = int(allboxs[i][3])
        color = (255, 0, 0)
        cv2.rectangle(im, (minX, minY), (maxX, maxY), color, 1)
    cv2.imencode(".jpg", im)[1].tofile(savepath)

def formatboxs(boxs):
    for i in range(len(boxs)):
        boxs[i] = ",".join(boxs[i])
    return boxs

if __name__ == '__main__':
    traindatapreparation(train_annotation)

    with open(train_annotation, encoding='utf-8') as f:
        train_lines = f.readlines()

    count = 0
    for data in train_lines:
        print(data)
        image, allboxs = getimage(data, num=5)
        line = data.split()
        savepath = 'E:\\code\\ssd\\ssd_02\\VOCdevkit\\test\\' + str(count) + '.jpg'
        image.save(savepath)
        count = count+1
        verify(savepath, allboxs)
        print(savepath)

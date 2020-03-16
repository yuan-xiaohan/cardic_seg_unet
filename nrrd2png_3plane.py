import cv2
import nrrd
import numpy as np
import os
import glob

# 读取文件并进行排序
def ContentSort(in_dir):
    in_content_name = os.listdir(in_dir)  # 返回目录下的所有文件和目录名
    # content_num = len(in_content_name)
    sort_num_first = []
    for file in in_content_name:
        sort_num_first.append(int((file.split("_")[2]).split(".")[0]))  # “BG_A_+序号.png” 根据 _ 分割，然后根据 . 分割，转化为数字类型
        sort_num_first.sort()
    sorted_file = []
    for sort_num in sort_num_first: #重新排序
        for file in in_content_name:
            if str(sort_num) == (file.split("_")[2]).split(".")[0]:
                sorted_file .append(file)
    return sorted_file

# Axial to Sagittal
def AxialToSagittal(a):
    s = a.transpose((2, 0, 1))
    s = s[::-1, :, :]
    return s

# Axial to Coronal
def AxialToCoronal(a):
    c = a.transpose((2,1,0))
    c = c[::-1,:,:]
    return c

# 映射到0~1之间
def Normalization(hu_value):
    hu_min = np.min(hu_value)
    hu_max = np.max(hu_value)
    normal_value = (hu_value - hu_min) / (hu_max - hu_min)
    return normal_value

# 根据窗宽、窗位计算出窗的最大值和最小值
def windowAdjust(img,ww,wl):
    win_min = wl - ww / 2
    win_max = wl + ww / 2
    # 根据窗最大值、最小值来截取img
    img_new = np.clip(img,win_min,win_max)
    return img_new

# 背景顺时针90度旋转 + 水平镜像
def ImgTrasform(img):
    row, col = img.shape[:2]
    M = cv2.getRotationMatrix2D((col / 2, row / 2), -90, 1)
    img_new = cv2.flip(cv2.warpAffine(img, M, (col, row)), 1)
    return img_new

# png to 3D mat
def png2mat(out_dir,sorted_file):
    num = len(sorted_file)
    matrix = np.zeros((512, 512, num), dtype=np.uint8)
    n = 0
    for in_name in sorted_file:
        in_content_path = os.path.join(out_dir, in_name)
        matrix[:, :, n] = cv2.imread(in_content_path)[:, :, 1]
        n = n + 1
    return matrix

def changenum(i):
    if i < 10:
        j = '000' + str(i)
    elif (i > 9 and i < 100):
        j = '00' + str(i)
    else:
        j = '0' + str(i)
    return j

# 3D mat to png
def mat2png(mat,title,out_dir):
    for index in range(mat.shape[2]):
        img = mat[:, :, index]
        cv2.imwrite(out_dir + title + changenum(index+1) + ".png", img)  # 命名为“BG_+序号.png”


# Start
root_nrrd = "E:\WHS_10_nrrd" #
root_png = "E:\WHS_10_png" #

pname = "huxiaoying" #病人名字
pnumber = "02" #病人序号

time = "30" #时相 !
bg_name = "11 Coro  DS_CorCTA  0.75  Bv36  3  30%.nrrd" #背景nrrd名字 !

#champer_name_list = glob.glob(os.path.join(root_nrrd,pname,time) + "\??-label.nrrd")

# 背景
in_bg_dir = os.path.join(root_nrrd,pname,time,bg_name) # 背景nrrd路径
# os.mkdir(os.path.join(root_png,pname)) # 新建文件夹
os.mkdir(os.path.join(root_png,pname,time)) # 新建文件夹

os.mkdir(os.path.join(root_png, pname, time, "BG"))  # 新建背景文件夹
os.mkdir(os.path.join(root_png, pname, time, "BG", "A"))  # 新建A轴文件夹
os.mkdir(os.path.join(root_png, pname, time, "BG", "S"))  # 新建S轴文件夹
os.mkdir(os.path.join(root_png, pname, time, "BG", "C"))  # 新建C轴文件夹
out_bg_a_dir = os.path.join(root_png, pname, time, "BG", "A")
out_bg_s_dir = os.path.join(root_png, pname, time, "BG", "S")
out_bg_c_dir = os.path.join(root_png, pname, time, "BG", "C")

readdata_bg, header_bg = nrrd.read(in_bg_dir)

# nrrd2png
for index in range(readdata_bg.shape[2]):
    map_bg = readdata_bg[:, :, index]
    map_bg = ImgTrasform(map_bg)
    map_bg = Normalization(windowAdjust(map_bg, 800, 200)) * 255
    cv2.imwrite(out_bg_a_dir + "\\" + pnumber + "_" + time + "_" + "BG" + "_A_" + changenum(index + 1) + ".png", map_bg)  # 命名为“BG_A_+序号.png”

# axial to sagittal and coronal
# sorted_bg = ContentSort(out_bg_a_dir)
sorted_bg = os.listdir(out_bg_a_dir)
bg_a = png2mat(out_bg_a_dir, sorted_bg)
bg_s = AxialToSagittal(bg_a)
mat2png(bg_s, "\\" + pnumber + "_" + time + "_" + "BG" + "_S_", out_bg_s_dir)
bg_c = AxialToCoronal(bg_a)
mat2png(bg_c, "\\" + pnumber + "_" + time + "_" + "BG" + "_C_", out_bg_c_dir)

# mask
#新建四腔室LA,LV,RA,RV
for champer in ["LA","LV","RA","RV"]:
    in_mask_dir = os.path.join(root_nrrd, pname, time) + "\\" + champer + "-label.nrrd"  # 掩膜nrrd路径
    os.mkdir(os.path.join(root_png, pname, time, champer))  # 新建掩膜文件夹
    os.mkdir(os.path.join(root_png, pname, time, champer, "A")) #新建A轴文件夹
    os.mkdir(os.path.join(root_png, pname, time, champer, "S"))  # 新建S轴文件夹
    os.mkdir(os.path.join(root_png, pname, time, champer, "C"))  # 新建C轴文件夹
    out_mask_a_dir = os.path.join(root_png, pname, time, champer, "A")
    out_mask_s_dir = os.path.join(root_png, pname, time, champer, "S")
    out_mask_c_dir = os.path.join(root_png, pname, time, champer, "C")


    readdata_mask, header_mask = nrrd.read(in_mask_dir)

    # nrrd2png
    for index in range(readdata_mask.shape[2]):
        map_mask = readdata_mask[:, :, index]
        ret, map_mask = cv2.threshold(map_mask, 0, 255, cv2.THRESH_BINARY)  # 二值化
        map_mask = ImgTrasform(map_mask)
        cv2.imwrite(out_mask_a_dir + "\\" + pnumber + "_" + time + "_" + champer + "_A_" + changenum(index + 1) + ".png", map_mask)  # 命名为“LV_A_+序号.png”

    # axial to sagittal and coronal
    # sorted_mask = ContentSort(out_mask_a_dir)
    sorted_mask = os.listdir(out_mask_a_dir)
    mask_a = png2mat(out_mask_a_dir, sorted_mask)
    mask_s = AxialToSagittal(mask_a)
    mat2png(mask_s, "\\" + pnumber + "_" + time + "_" + champer + "_S_", out_mask_s_dir)
    mask_c = AxialToCoronal(mask_a)
    mat2png(mask_c, "\\" + pnumber + "_" + time + "_" + champer + "_C_", out_mask_c_dir)



'''

in_bg_dir = "E:\WHS_10_nrrd\chnxiaoqing_label\\10\\9 Func  DS_CorCTA  0.75  Bv36  3  10%.nrrd" # 背景nrrd路径
os.mkdir("E:\WHS_10_png\chnxiaoqing\\10") # 新建文件夹
os.mkdir("E:\WHS_10_png\chnxiaoqing\\10\BG_A") # 新建背景文件夹(A)
os.mkdir("E:\WHS_10_png\chnxiaoqing\\10\BG_S") # 新建背景文件夹(S)
os.mkdir("E:\WHS_10_png\chnxiaoqing\\10\BG_C") # 新建背景文件夹(C)
out_bg_a_dir = "E:\WHS_10_png\chnxiaoqing\\10\BG_A"
out_bg_s_dir = "E:\WHS_10_png\chnxiaoqing\\10\BG_S"
out_bg_c_dir = "E:\WHS_10_png\chnxiaoqing\\10\BG_C"

in_mask_dir = "E:\WHS_10_nrrd\chnxiaoqing_label\\10\\LV-label.nrrd" # 掩膜nrrd路径
os.mkdir("E:\WHS_10_png\chnxiaoqing\\10\LV_A") # 新建掩膜文件夹(A)
os.mkdir("E:\WHS_10_png\chnxiaoqing\\10\LV_S") # 新建掩膜文件夹(S)
os.mkdir("E:\WHS_10_png\chnxiaoqing\\10\LV_C") # 新建掩膜文件夹(C)
out_mask_a_dir = "E:\WHS_10_png\chnxiaoqing\\10\LV_A"
out_mask_s_dir = "E:\WHS_10_png\chnxiaoqing\\10\LV_S"
out_mask_c_dir = "E:\WHS_10_png\chnxiaoqing\\10\LV_C"


readdata_bg, header_bg = nrrd.read(in_bg_dir)
readdata_mask, header_mask = nrrd.read(in_mask_dir)


# nrrd2png
# bg
for index in range(readdata_bg.shape[2]):
    map_bg = readdata_bg[:,:,index]
    map_bg = ImgTrasform(map_bg)
    map_bg = Normalization(windowAdjust(map_bg, 800, 200))*255
    cv2.imwrite(out_bg_a_dir + "\BG_A_" + str(index+1) + ".png", map_bg)  # 命名为“BG_A_+序号.png”
# mask
for index in range(readdata_mask.shape[2]):
    map_mask = readdata_mask[:,:,index]
    ret,map_mask = cv2.threshold(map_mask, 0, 255, cv2.THRESH_BINARY) #二值化
    map_mask = ImgTrasform(map_mask)
    cv2.imwrite(out_mask_a_dir + "\LV_A_" + str(index+1) + ".png", map_mask)  # 命名为“LV_A_+序号.png”

# axial to sagittal and coronal
# bg
sorted_bg = ContentSort(out_bg_a_dir)
bg_a = png2mat(out_bg_a_dir,sorted_bg)
bg_s = AxialToSagittal(bg_a)
mat2png(bg_s,"\BG_S_",out_bg_s_dir)
bg_c = AxialToCoronal(bg_a)
mat2png(bg_c,"\BG_C_",out_bg_c_dir)

# mask
sorted_mask = ContentSort(out_mask_a_dir)
mask_a = png2mat(out_mask_a_dir,sorted_mask)
mask_s = AxialToSagittal(mask_a)
mat2png(mask_s,"\LV_S_",out_mask_s_dir)
mask_c = AxialToCoronal(mask_a)
mat2png(mask_c,"\LV_C_",out_mask_c_dir)

'''
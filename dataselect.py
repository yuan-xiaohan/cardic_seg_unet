import os
import shutil

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

mask_src="E:\\yuanxiaohan\\Cardic_segmentation\\data\\LV\\1526993_45\LV_A"
img_src="E:\\yuanxiaohan\\Cardic_segmentation\\data\\LV\\1526993_45\BG_A"
mask_dst="E:.\\data\\train\masks"
img_dst="E:.\\data\\train\images"
mask = ContentSort(mask_src)
img = ContentSort(img_src)
mask_select_index = mask[63:146]
img_select_index = img[63:146]

print(len(img_select_index))
for i in img_select_index:
    shutil.copy(os.path.join(img_src,str(i)), img_dst)


for i in mask_select_index:
    shutil.copy(os.path.join(mask_src,str(i)), mask_dst)


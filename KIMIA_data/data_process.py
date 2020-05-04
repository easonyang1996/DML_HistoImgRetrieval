import cv2
import os

train_ori = '../KIMIAPath24_RGB/Training/'
test_ori = '../KIMIAPath24_RGB/RGBPatches-test/'

train_new = './train/'
test_new = './test/'

path_ori = [train_ori, test_ori]
path_new = [train_new, test_new]

for i in range(2):
    for root, dirs, files in os.walk(path_ori[i]):
        if files == []:
            for d in dirs:
                os.makedirs(os.path.join(path_new[i],d))
        if dirs == []:
            category = root.split('/')[-1]
            cate_path = os.path.join(path_new[i], category)
            print(cate_path)
            for f in files:
                name = category + '_' + f.split('.')[0] + '.png'
                img = cv2.imread(os.path.join(root,f))
                resize_img = cv2.resize(img, (300,300), interpolation=cv2.INTER_CUBIC)
                #print(cate_path, name)
                cv2.imwrite(os.path.join(cate_path, name), resize_img)
                 

             

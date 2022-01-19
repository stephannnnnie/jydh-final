import lpips
import cv2
import torchvision.transforms as transforms
import glob
import os
import numpy as np
import torch


if __name__=='__main__':
    img_data_dir='./siggraph'
    paths = glob.glob(os.path.join(img_data_dir, '*.png'))
    mask_paths = []
    img_paths = []
    d=[]
    for path in paths:
        if len(os.path.splitext(os.path.basename(path))[0].split('_'))==1:
            mask_file = os.path.splitext(os.path.basename(path))[0] + '_siggraph17.png'
            print(mask_file)
            mask_path = os.path.join(img_data_dir, mask_file)
            mask_paths.append(mask_path)
            img_paths.append(path)
    for i in range(len(mask_paths)):
        img1=cv2.imread(mask_paths[i])
        img2=cv2.imread(img_paths[i])
        transf = transforms.ToTensor()
        img1_tensor = transf(img1)
        img2_tensor = transf(img2)
        loss_fn = lpips.LPIPS(net='alex')
        d.append(np.min(loss_fn(img1_tensor,img2_tensor).detach().numpy()))
    print(np.mean(d))

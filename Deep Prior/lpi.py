import lpips
import cv2
import torchvision.transforms as transforms

if __name__=='__main__':
    img1=cv2.imread('./images/2/1_colored_samples.png')
    img2=cv2.imread('./images/2/1_real_samples.png')
    print(type(img1))
    transf = transforms.ToTensor()
    img1_tensor = transf(img1)
    img2_tensor = transf(img2)
    loss_fn = lpips.LPIPS(net='alex')
    d = loss_fn(img1_tensor,img2_tensor)
    print(d)
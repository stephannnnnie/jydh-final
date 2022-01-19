
import argparse
import matplotlib.pyplot as plt
import glob
import os

from colorizers import *

parser = argparse.ArgumentParser()
parser.add_argument('-i','--img_path', type=str, default='./test')
parser.add_argument('--use_gpu', action='store_true', help='whether to use GPU')
parser.add_argument('-o','--save_prefix', type=str, default='saved', help='will save into this file with {eccv16.png, siggraph17.png} suffixes')
opt = parser.parse_args()

# load colorizers
colorizer_eccv16 = eccv16(pretrained=True).eval()
colorizer_siggraph17 = siggraph17(pretrained=True).eval()
if(opt.use_gpu):
	colorizer_eccv16.cuda()
	colorizer_siggraph17.cuda()

# default size to process images is 256x256
# grab L channel in both original ("orig") and resized ("rs") resolutions
img_paths = glob.glob(os.path.join(opt.img_path,"*.jpg"))
for path in img_paths:
	print(path)
	img = load_img(path)
	(tens_l_orig, tens_l_rs) = preprocess_img(img, HW=(256,256))
	if(opt.use_gpu):
		tens_l_rs = tens_l_rs.cuda()

	# colorizer outputs 256x256 ab map
	# resize and concatenate to original L channel
	img_bw = postprocess_tens(tens_l_orig, torch.cat((0*tens_l_orig,0*tens_l_orig),dim=1))
	out_img_eccv16 = postprocess_tens(tens_l_orig, colorizer_eccv16(tens_l_rs).cpu())
	out_img_siggraph17 = postprocess_tens(tens_l_orig, colorizer_siggraph17(tens_l_rs).cpu())

	plt.imsave('./eccv/%s_eccv16.png'%os.path.splitext(os.path.basename(path))[0].split('_')[0], out_img_eccv16)
	plt.imsave('./siggraph/%s_siggraph17.png'%os.path.splitext(os.path.basename(path))[0].split('_')[0], out_img_siggraph17)

# plt.figure(figsize=(12,8))
# plt.subplot(2,2,1)
# plt.imshow(img)
# plt.title('Original')
# plt.axis('off')
#
# plt.subplot(2,2,2)
# plt.imshow(img_bw)
# plt.title('Input')
# plt.axis('off')
#
# plt.subplot(2,2,3)
# plt.imshow(out_img_eccv16)
# plt.title('Output (ECCV 16)')
# plt.axis('off')
#
# plt.subplot(2,2,4)
# plt.imshow(out_img_siggraph17)
# plt.title('Output (SIGGRAPH 17)')
# plt.axis('off')
# plt.show()

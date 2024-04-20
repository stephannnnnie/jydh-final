# README

Here is the final project of CS308 Computer Vision in SUSTech. We chose *ColoringWith Limited Data:*

*Few-Shot Colorization via Memory-Augmented Networks method to do colorization. Our main works are:



1. Train MemoPainter and compare results with the pre-trained model
2. Ablation experiment of Memory network
3. Changing the generator to FCN to see how well the memory network works
4. Turning parameters to evaluate the robustness of MemoPainter
5. Analysis of Threshold Triplet Loss (TTL)
6. Compare results of MemoPainter to CIC and Deep Prior



The dataset we used is Pokemon, which can be found at https://www.kaggle.com/kvpratama/pokemon-images-dataset.



## MemoPainter

We first got recured codes from https://github.com/dongheehand/MemoPainter-PyTorch. You can use mode_with_memory.py to train the original MemoPainter and FCN generator MemoPainter (change generator as follows). Training lines are in train.sh.

```python
# generator = unet_generator(args.input_channel, args.output_channel, args.n_feats, args.color_feat_dim)
generator = VGG16(args.input_channel, args.output_channel, args.n_feats, args.color_feat_dim).to(device)
```

Also, you can use mode.py to train the network without a memory network. Here are some of our visualization results.

![image-20220119094912177](https://github.com/stephannnnnie/jydh-final/blob/main/images/image-20220118103920239.png)

![image-20220119094937401](https://github.com/stephannnnnie/jydh-final/blob/main/images/image-20220118012935113.png)

### LPIPS

Memtioned that there are no quantitative evaluation, so we added LPIPS part in mode.py and mode_with_memory.py. If you just want to quantify your results, you can use loss.py by puting your results and GT images in one document and change the path in loss.py. 

### New testing data

We found that results of training and results of pretrained model both did a fine job, but sometimes colorized images of pretrained model are far better than the colorized images of training model. As parameters are the same, we thought here may exits an problem that the pretrained model may be trained by other parts of the dataset, which may includes our testing data. Thus, we made a new testing dataset using images that are found on the internet. You can get that from test.zip.



## CIC

You can see original codes in https://github.com/richzhang/colorization. We only used pretrained model to test our images.



## Deep Prior

You can see original codes in https://github.com/inkImage/real-time-user-guided-colorization_pytorch. We had adjust parameters and codes to suit our dataset.Here are example lines of training and testing.

```
python deep_color.py
```

```
python sampling.py --model unet100.pkl
```

You can choose any model you want while testing.



## Comparison

![image-20220119101511865](https://github.com/stephannnnnie/jydh-final/blob/main/images/image-20220118200020037.png)


# Session 14-15 - RCNN

###	Objective
Create a custom dataset for monocular depth estimation and segmentation simultaneously.

### Dataset Creation

#### Background (bg)
 - "scene" images. Like the front of shops, etc.
 - 100 images of streets were downloaded from the internet.
 - Each image was resized to 224 x 224
 - Number of images: 100
 - Image dimensions: (224, 224, 3)
 - Directory size: 2.5M
 - Mean: [0.5039, 0.5001, 0.4849]
 - Std: [0.2465, 0.2463, 0.2582]

<img src="images/bg.png">

#### Foreground (fg)
 - Images of objects with transparent background
 - 100 images of footballers were downloaded from the internet.
 - Using GIMP, the foreground was cutout. and the background was made transparent by adding an alpha layer.
 - Each image was rescaled to keep height 105 and resizing width while maintaining aspect ratio.
 - Number of images: 100
 - Image dimensions: (105, width, 4)
 - Directory size: 1.2M

<img src="images/fg.png">

#### Foreground Mask (fg_mask)
 - For every foreground its corresponding mask was created
 - Using GIMP, the foreground was filled with white and the background was filled with black.
 - Image was stored as a grayscale image.
 - Each image was rescaled to keep height 105 and resizing width while maintaining aspect ratio.
 - Number of images: 100
 - Image dimensions: (105, width)
 - Directory size: 404K

<img src="images/fg_mask.png">

#### Foreground Overlayed on Background (fg_bg)
 - For each background
	 - Overlay each foreground randomly 20 times on the background
	 - Flip the foreground and again overlay it randomly 20 times on the background
 - Number of images: 100\*100\*2\*20 = 400,000
 - Image dimensions: (224, 224, 3)
 - Directory size: 4.2G
 - Mean: [0.5056, 0.4969, 0.4817]
 - Std: [0.2486, 0.2490, 0.2604]

<img src="images/fg_bg.png">

#### Foreground Overlayed on Background Mask (fg_bg_mask)
 - For every foreground overlayed on background, its corresponding mask was created.
 - The mask was created by pasting the foreground mask on a black image at the same position the foreground was overlayed.
 -  Image was stored as a grayscale image.
 - Number of images: 400,000
 - Image dimensions: (224, 224)
 - Directory size: 1.6G
 - Mean: [0.0454]
 - Std: [0.2038]

<img src="images/fg_bg_mask.png">

#### Foreground Overlayed on Background Depth Map (fg_bg_depth)
 - For every foreground overlayed on background, its corresponding depth map was generated.
 - A pre-trained monocular depth estimation model [DenseDepth](https://github.com/ialhashim/DenseDepth/blob/master/DenseDepth.ipynb) was used to generate the depth maps.
 - Image was stored as a grayscale image.
 - Number of images: 400,000
 - Image dimensions: (224, 224)
 - Directory size: 1.6G
 - Mean: [0.4334]
 - Std: [0.2715]

<img src="images/fg_bg_depth.png">

### Dataset Statistics

| Type | Dimensions | Mean | Std |
|---|---|---|---|
| **Background** | (224,224,3) | (0.5039, 0.5001, 0.4849) | (0.2465, 0.2463, 0.2582) |
| **Foreground-Background** | (224,224,3) | (0.5056, 0.4969, 0.4817) | (0.2486, 0.2490, 0.2604) |
| **Foreground-Background Mask** | (224,224) | (0.0454) | (0.2038) |
| **Foreground-Background Depth** | (224,224) | (0.4334) | (0.2715) |

### Dataset Link

 - Link: https://drive.google.com/file/d/1KY-6ndddnDSXTp974YeubFKEMTbKmqiH/view?usp=sharing
 - Size:
	 - Zip: 5G
	 - Unzip: 7.3G 

| Type | Count | size |
|---|---|---|
| **Background** | 100 | 2.5M |
| **Foreground** | 100 | 1.2M |
| **Foreground Mask** | 100 | 404K |
| **Foreground-Background** | 400,000 | 4.2G |
| **Foreground-Background Mask** | 400,000 | 1.6G |
| **Foreground-Background Depth** | 400,000 | 1.6G |
| **Total** | 1,200,300 | 7.3G |


### Dataset Visualization
<img src="images/dataset.png">

### Resources

 - Code to overlay foreground on background and corresponding masks: 
	 - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S14-15/EVA4_S15A_gen_fg_bg.ipynb)
 - Code to generate depth maps for foreground overlayed on background: 
	 - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S14-15/EVA4_S15A_gen_fg_bg_depth_maps.ipynb)
	 - [Forked Repo](https://github.com/uday96/DenseDepth/tree/cars_fg)
 - Code to compute the combine the dataset and analyse the statistics:
	 - [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S14-15/EVA4_S15A_data_statistics.ipynb)

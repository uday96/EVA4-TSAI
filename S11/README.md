
# Session 11 - Super Convergence

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S11/EVA4_S11_Solution.ipynb)

###	Objective
Achieve an accuracy of **90%** on the **CIFAR-10** dataset using:

- Custom ResNet architecture
- One Cycle Policy:
	- Total Epochs = 24
	-  Max at Epoch = 5
	-  LRMIN = FIND
	-  LRMAX = FIND
	-  NO Annihilation
- Batch Size: 512
- Image Augmentation:
	- RandomCrop(32, 32) (after padding of 4)
	- FlipLR
	- CutOut(8, 8)

Simulate the Cyclic LR schedules and visualize the cyclic traingular plot.

#### Model Architecture

    1.  PrepLayer - Conv 3x3 s1, p1) >> BN >> RELU [64k]
    2.  Layer1 -
        1.  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [128k]
        2.  R1 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [128k]
        3.  Add(X, R1)
    3.  Layer 2 -
        1.  Conv 3x3 [256k]
        2.  MaxPooling2D
        3.  BN
        4.  ReLU
    4.  Layer 3 -
        1.  X = Conv 3x3 (s1, p1) >> MaxPool2D >> BN >> RELU [512k]
        2.  R2 = ResBlock( (Conv-BN-ReLU-Conv-BN-ReLU))(X) [512k]
        3.  Add(X, R2)
    5.  MaxPooling with Kernel Size 4
    6.  FC Layer
    7.  SoftMax

###  Parameters and Hyperparameters
- Loss Function: Negative Log Likelihood
- Optimizer: SGD
- Scheduler: OneCycleLR
- Batch Size: 512
- Epochs: 24
- L1 decay: 2e-6
- L2 decay: 6e-4

### Image Augmentation Techniques
- Pad: h=40, w=40, p=1.0
- OneOf: p=1.0
	- RandomCrop: h=32, w=32, p=0.8
	- CenterCrop: h=32, w=32, p=0.2
- Horizontal Flip: p=0.5
- Coarse Dropout: holes=1, h=8, w=8, p=0.75

### Learning Rate Parameters
OneCycleLR:
- max_lr: 0.01682
- div_factor: 10
- final_div_factor: 1
- anneal_strategy: linear
- pct: 5/24

### Results
Achieved  an accuracy of **90.62%** in 24th epoch.

#### LR Range Test
- Best Accuracy: 89.37
- Best Learning Rate: 0.016820061224489796

<img src="images/lr_range_test.png">

#### Change in Learning Rate
<img src="images/lr_change.png">

#### Change in Loss
<img src="images/loss_change.png">

#### Change in Accuracy
<img src="images/accuracy_change.png">

### GradCAM Visualizations

Visualize GradCAM at different convolutional layers to understand where the network is looking at while prediction.

<img src="images/gradcam_incorrect_0_tdog_pcat.png">

<img src="images/gradcam_incorrect_1_tcat_pdeer.png">

<img src="images/gradcam_incorrect_2_tcar_pplane.png">

<img src="images/gradcam_incorrect_3_thorse_pcat.png">

<img src="images/gradcam_incorrect_4_thorse_pship.png">

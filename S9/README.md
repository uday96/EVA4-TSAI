
# Session 9 - Data Augmentation and GradCAM

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S9/EVA4_S9_Solution.ipynb)

###	Objective
Achieve an accuracy of **87%** on the **CIFAR-10** dataset using **ResNet18**:

-  Use Albumentations for image augmenation
- Implement Grad-CAM: Gradient-weighted Class Activation Mapping

###  Parameters and Hyperparameters

- Loss Function: Cross Entropy Loss
- Optimizer: SGD
- Scheduler: OneCycleLR
- Batch Size: 64
- Learning Rate: lr=0.1, max_lr=0.01
- Epochs: 35
- Dropout: 0.15
- L1 decay: 1e-6
- L2 decay: 1e-3

### Image Augmentation Techniques

- Horizontal Flip: p=0.5
- Hue Saturation Value: p=0.25
- Rotate: limit=15, p=0.5
- Coarse Dropout: holes=1, h=(4,16), w=(4,16), p=0.75

### Results
Achieved  an accuracy of **92.14%** in 33rd epoch.

#### Validation Loss
<img src="images/val_loss_change.png">

#### Validation Accuracy
<img src="images/val_accuracy_change.png">

#### Correctly Classified Images
<img src="images/correct_imgs.png">

#### Misclassified Images
<img src="images/misclassified_imgs.png">

### GradCAM Visualizations

Visualize GradCAM at different convolutional layers to understand where the network is looking at while prediction.

#### Correctly Classified Images

GradCAM at different convolutional layers for the class: *Plane*
<img src="images/gradcam_correct_0_plane.png">
*Image: Actual: Plane, Predicted: Plane*

GradCAM at different convolutional layers for the class: *Bird*
<img src="images/gradcam_correct_1_bird.png">
*Image: Actual: Bird, Predicted: Bird*

GradCAM at different convolutional layers for the class: *Bird*
<img src="images/gradcam_correct_2_bird.png">
*Image: Actual: Bird, Predicted: Bird*

GradCAM at different convolutional layers for the class: *Truck*
<img src="images/gradcam_correct_3_truck.png">
*Image: Actual: Truck, Predicted: Truck*

GradCAM at different convolutional layers for the class: *Truck*
<img src="images/gradcam_correct_4_truck.png">
*Image: Actual: Truck, Predicted: Truck*

#### Misclassified Images

GradCAM at different convolutional layers for the class: *Truck*
<img src="images/gradcam_incorrect_0_truck.png">
*Image: Actual: Ship, Predicted: Truck*

GradCAM at different convolutional layers for the class: *Dog*
<img src="images/gradcam_incorrect_1_dog.png">
*Image: Actual: Cat, Predicted: Dog*

GradCAM at different convolutional layers for the class: *Truck*
<img src="images/gradcam_incorrect_2_truck.png">
*Image: Actual: Plane, Predicted: Truck*

GradCAM at different convolutional layers for the class: *Plane*
<img src="images/gradcam_incorrect_3_plane.png">
*Image: Actual: Deer, Predicted: Plane*

GradCAM at different convolutional layers for the class: *Dog*
<img src="images/gradcam_incorrect_4_dog.png">
*Image: Actual: Bird, Predicted: Dog*


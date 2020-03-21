

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

##### GradCAM at different convolutional layers for the class: *Plane*

| Truth: Plane, Predicted: Plane |
|---|
| <img src="images/gradcam_correct_0_plane.png">  |

##### GradCAM at different convolutional layers for the class: *Bird*

| Truth: Bird, Predicted: Bird |
|---|
| <img src="images/gradcam_correct_1_bird.png"> |

##### GradCAM at different convolutional layers for the class: *Bird*

| Truth: Bird, Predicted: Bird |
|---|
| <img src="images/gradcam_correct_2_bird.png"> |

##### GradCAM at different convolutional layers for the class: *Truck*

| Truth: Truck, Predicted: Truck |
|---|
| <img src="images/gradcam_correct_3_truck.png"> |

##### GradCAM at different convolutional layers for the class: *Truck*

| Truth: Truck, Predicted: Truck |
|---|
| <img src="images/gradcam_correct_4_truck.png"> |

#### Misclassified Images

##### GradCAM at different convolutional layers for the class: *Truck*

| Truth: Ship, Predicted: Truck |
|---|
| <img src="images/gradcam_incorrect_0_truck.png"> |

##### GradCAM at different convolutional layers for the class: *Dog*

| Truth: Cat, Predicted: Dog |
|---|
| <img src="images/gradcam_incorrect_1_dog.png"> |

##### GradCAM at different convolutional layers for the class: *Truck*

| Truth: Plane, Predicted: Truck |
|---|
| <img src="images/gradcam_incorrect_2_truck.png"> |

##### GradCAM at different convolutional layers for the class: *Plane*

| Truth: Deer, Predicted: Plane |
|---|
| <img src="images/gradcam_incorrect_3_plane.png"> |

##### GradCAM at different convolutional layers for the class: *Dog*

| Truth: Bird, Predicted: Dog |
|---|
| <img src="images/gradcam_incorrect_4_dog.png"> |

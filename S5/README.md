
# Session 5 - MNIST 99.4% Test Accuracy

###	Objective
Achieve an accuracy of **99.4%** on the **MNIST** dataset with the following constraints:

-	99.4% validation accuracy
-	Less than 10k Parameters
-  Less than 15 Epochs
-	No fully connected layers

### Code 1

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S5/EVA4_S5_Solution_F1.ipynb)

#### Target

-   Initial working setup
-   Fix the model skeleton/structure
    -   Add layers until receptive field reaches image size (28)
    -   Use AvgPool in the output layer
    -   Add computation after AvgPool so that we dont force the previous channels to act as one-hot vectors

#### Result

-   Parameters: 11,824
-   Best Train Accuracy: 99.36%
-   Best Test Accuracy: 99.10%

#### Analysis

-   Model is working
-   Model is over-fitting
-   Need to reduce the number of parameters to under 10,000

### Code 2

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S5/EVA4_S5_Solution_F2.ipynb)

#### Target

-   Make the model lighter by changing the number of channels

#### Result

-   Parameters: 8,368
-   Best Train Accuracy: 99.26%
-   Best Test Accuracy: 99.13%

#### Analysis

-   Model is slightly over-fitting
-   Model can be pushed further
-   Number of parameters is under 10,000

### Code 3

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S5/EVA4_S5_Solution_F3.ipynb)

#### Target

-   Add Batch-Noramlization to improve the efficiency of the model

#### Result

-   Parameters: 8,500
-   Best Train Accuracy: 99.52%
-   Best Test Accuracy: 99.41%

#### Analysis

-   Good model
-   Batch-Norm has improved the model efficiency. The best train accuracy improved from 99.26% to 99.52%
-   Model is slightly over-fitting
-   Not seeing 99.4% as often as we want to

### Code 4

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S5/EVA4_S5_Solution_F4.ipynb)

#### Target

-   Add dropout to handle overfitting

#### Result

-   Parameters: 8,500
-   Best Train Accuracy: 99.32%
-   Best Test Accuracy: 99.46%

#### Analysis

-   No overfitting at all
-   Not seeing 99.4% as often as we want to
-   The test accuracy keeps fluctuating possibly because of a high learning rate after few epochs

### Code 5

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/uday96/EVA4-TSAI/blob/master/S5/EVA4_S5_Solution_F5.ipynb)

#### Target

-   Add LR Scheduler
-   In Code 4 we observed that the accuracy dropped after 6th epoch. So, use StepLR with step set to 6

#### Result

-   Parameters: 8,500
-   Best Train Accuracy: 99.31%
-   Best Test Accuracy: 99.49%

#### Analysis

-   Good model
-   Reached target accuracy in 8th epoch (99.46%)
-   Reaching the target accuracy consistently

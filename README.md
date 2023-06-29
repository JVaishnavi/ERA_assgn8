# Comparison of Batch Normalisation vs Layer Normalisation vs Group Normalisation

We have used CIFAR10 dataset for the purpose of comparison between different normalisation techniques. 
CIFAR10 dataset contains images of 32x32x3 and sample images look like:

![output_10_1](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/0307c384-9ed8-4451-a9ca-54158a447c28)

## How to read the files

model_BN.py contains the pytorch model for applying batch normalisation to the CIFAR-10 dataset.
model_LN.py contains the pytorch model for applying batch normalisation to the CIFAR-10 dataset.
model_GN.py contains the pytorch model for applying batch normalisation to the CIFAR-10 dataset.
utils.py contains all the utility function for the current CNN architecture

notebook_BN.ipynb contains the notebook for running the CNN model with batch normalisation from model_BN.py file
notebook_LN.ipynb contains the notebook for running the CNN model with batch normalisation from model_LN.py file
notebook_GN.ipynb contains the notebook for running the CNN model with batch normalisation from model_GN.py file

## Normalisation

Normalisation is a pre-processing technique used to standardize data. It is a procedure to change the value of the numeric variable in the dataset to a typical scale, without misshaping contrasts in the range of value. There are different kinds of normalisation:
* Batch Normalisation
* Layer Normalisation
* Group Normalisation
* Instance Normalisation
  
![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/b187ca98-34b2-485f-b4f9-8bbaa60efb7c)


## Batch Normalisation:

Batch Norm is a normalisation technique done between the layers of a Neural Network instead of in the raw data. It is done along mini-batches instead of the full data set. It serves to speed up training and use higher learning rates, making learning easier. BBatch Normalisation solves a problem called the Internal Covariate shift.

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/41359c1d-f225-4229-a08a-c877111ee2b3)

## Layer Normalisation:

Unlike batch normalisation, Layer Normalisation directly estimates the normalisation statistics from the summed inputs to the neurons within a hidden layer so the normalisation does not introduce any new dependencies between training cases.

We compute the layer normalisation statistics over all the hidden units in the same layer as follows:

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/f8dd87da-ab11-4891-8502-8868be09ce3a)

where &Eta; denotes the number of hidden units in a layer. Under layer normalisation, all the hidden units in a layer share the same normalisation terms &mu; and &sigma;, but different training cases have different normalisation terms. Unlike batch normalisation, layer normalisation does not impose any constraint on the size of the mini-batch and it can be used in the pure online regime with batch size 1.

## Group Normalisation:

Group Normalisation is a normalisation layer that divides channels into groups and normalizes the features within each group. GN does not exploit the batch dimension, and its computation is independent of batch sizes.

Group Normalisation is defined as:

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/a4405346-2771-44d4-ae56-42f310c96c8f)

Here $x$ is the feature computed by a layer, and $i$ is an index. Formally, a Group Norm layer computes &mu; and &sigma; in a set $S_i$ that covers all the groups.

## Comparison

### Model summary
Below is the model summary for all three models.

#### Batch normalisation

```python
get_summary(Net)
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 16, 32, 32]             432
                  ReLU-2           [-1, 16, 32, 32]               0
           BatchNorm2d-3           [-1, 16, 32, 32]              32
               Dropout-4           [-1, 16, 32, 32]               0
                Conv2d-5           [-1, 24, 32, 32]           3,456
                  ReLU-6           [-1, 24, 32, 32]               0
           BatchNorm2d-7           [-1, 24, 32, 32]              48
               Dropout-8           [-1, 24, 32, 32]               0
                Conv2d-9            [-1, 8, 32, 32]             192
                 ReLU-10            [-1, 8, 32, 32]               0
          BatchNorm2d-11            [-1, 8, 32, 32]              16
              Dropout-12            [-1, 8, 32, 32]               0
            MaxPool2d-13            [-1, 8, 16, 16]               0
               Conv2d-14           [-1, 16, 16, 16]           1,152
                 ReLU-15           [-1, 16, 16, 16]               0
          BatchNorm2d-16           [-1, 16, 16, 16]              32
              Dropout-17           [-1, 16, 16, 16]               0
               Conv2d-18           [-1, 32, 16, 16]           4,608
                 ReLU-19           [-1, 32, 16, 16]               0
          BatchNorm2d-20           [-1, 32, 16, 16]              64
              Dropout-21           [-1, 32, 16, 16]               0
               Conv2d-22           [-1, 48, 16, 16]          13,824
                 ReLU-23           [-1, 48, 16, 16]               0
          BatchNorm2d-24           [-1, 48, 16, 16]              96
              Dropout-25           [-1, 48, 16, 16]               0
               Conv2d-26           [-1, 10, 16, 16]             480
                 ReLU-27           [-1, 10, 16, 16]               0
          BatchNorm2d-28           [-1, 10, 16, 16]              20
              Dropout-29           [-1, 10, 16, 16]               0
            MaxPool2d-30             [-1, 10, 8, 8]               0
               Conv2d-31             [-1, 16, 8, 8]           1,440
                 ReLU-32             [-1, 16, 8, 8]               0
          BatchNorm2d-33             [-1, 16, 8, 8]              32
              Dropout-34             [-1, 16, 8, 8]               0
               Conv2d-35             [-1, 32, 6, 6]           4,608
                 ReLU-36             [-1, 32, 6, 6]               0
          BatchNorm2d-37             [-1, 32, 6, 6]              64
              Dropout-38             [-1, 32, 6, 6]               0
               Conv2d-39             [-1, 64, 4, 4]          18,432
                 ReLU-40             [-1, 64, 4, 4]               0
          BatchNorm2d-41             [-1, 64, 4, 4]             128
              Dropout-42             [-1, 64, 4, 4]               0
            AvgPool2d-43             [-1, 64, 1, 1]               0
               Conv2d-44             [-1, 10, 1, 1]             640
    ================================================================
    Total params: 49,796
    Trainable params: 49,796
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 2.45
    Params size (MB): 0.19
    Estimated Total Size (MB): 2.65
    ----------------------------------------------------------------





    <function torchsummary.torchsummary.summary(model, input_size, batch_size=-1, device='cuda')>


#### Layer Normalisation



```python
get_summary(Net)
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 16, 32, 32]             432
                  ReLU-2           [-1, 16, 32, 32]               0
             LayerNorm-3           [-1, 16, 32, 32]           2,048
               Dropout-4           [-1, 16, 32, 32]               0
                Conv2d-5           [-1, 24, 32, 32]           3,456
                  ReLU-6           [-1, 24, 32, 32]               0
             LayerNorm-7           [-1, 24, 32, 32]           2,048
               Dropout-8           [-1, 24, 32, 32]               0
                Conv2d-9            [-1, 8, 32, 32]             192
                 ReLU-10            [-1, 8, 32, 32]               0
            LayerNorm-11            [-1, 8, 32, 32]           2,048
              Dropout-12            [-1, 8, 32, 32]               0
            MaxPool2d-13            [-1, 8, 16, 16]               0
               Conv2d-14           [-1, 16, 16, 16]           1,152
                 ReLU-15           [-1, 16, 16, 16]               0
            LayerNorm-16           [-1, 16, 16, 16]             512
              Dropout-17           [-1, 16, 16, 16]               0
               Conv2d-18           [-1, 24, 16, 16]           3,456
                 ReLU-19           [-1, 24, 16, 16]               0
            LayerNorm-20           [-1, 24, 16, 16]             512
              Dropout-21           [-1, 24, 16, 16]               0
               Conv2d-22           [-1, 32, 16, 16]           6,912
                 ReLU-23           [-1, 32, 16, 16]               0
            LayerNorm-24           [-1, 32, 16, 16]             512
              Dropout-25           [-1, 32, 16, 16]               0
               Conv2d-26           [-1, 10, 16, 16]             320
                 ReLU-27           [-1, 10, 16, 16]               0
            LayerNorm-28           [-1, 10, 16, 16]             512
              Dropout-29           [-1, 10, 16, 16]               0
            MaxPool2d-30             [-1, 10, 8, 8]               0
               Conv2d-31             [-1, 16, 8, 8]           1,440
                 ReLU-32             [-1, 16, 8, 8]               0
            LayerNorm-33             [-1, 16, 8, 8]             128
              Dropout-34             [-1, 16, 8, 8]               0
               Conv2d-35             [-1, 32, 6, 6]           4,608
                 ReLU-36             [-1, 32, 6, 6]               0
            LayerNorm-37             [-1, 32, 6, 6]              72
              Dropout-38             [-1, 32, 6, 6]               0
               Conv2d-39             [-1, 64, 4, 4]          18,432
                 ReLU-40             [-1, 64, 4, 4]               0
            LayerNorm-41             [-1, 64, 4, 4]              32
              Dropout-42             [-1, 64, 4, 4]               0
            AvgPool2d-43             [-1, 64, 1, 1]               0
               Conv2d-44             [-1, 10, 1, 1]             640
    ================================================================
    Total params: 49,464
    Trainable params: 49,464
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 2.26
    Params size (MB): 0.19
    Estimated Total Size (MB): 2.46
    ----------------------------------------------------------------





    <function torchsummary.torchsummary.summary(model, input_size, batch_size=-1, device='cuda')>


#### Group normalisation

```python
get_summary(Net)
```

    ----------------------------------------------------------------
            Layer (type)               Output Shape         Param #
    ================================================================
                Conv2d-1           [-1, 16, 32, 32]             432
                  ReLU-2           [-1, 16, 32, 32]               0
             GroupNorm-3           [-1, 16, 32, 32]              32
               Dropout-4           [-1, 16, 32, 32]               0
                Conv2d-5           [-1, 24, 32, 32]           3,456
                  ReLU-6           [-1, 24, 32, 32]               0
             GroupNorm-7           [-1, 24, 32, 32]              48
               Dropout-8           [-1, 24, 32, 32]               0
                Conv2d-9            [-1, 8, 32, 32]             192
                 ReLU-10            [-1, 8, 32, 32]               0
            GroupNorm-11            [-1, 8, 32, 32]              16
              Dropout-12            [-1, 8, 32, 32]               0
            MaxPool2d-13            [-1, 8, 16, 16]               0
               Conv2d-14           [-1, 16, 16, 16]           1,152
                 ReLU-15           [-1, 16, 16, 16]               0
            GroupNorm-16           [-1, 16, 16, 16]              32
              Dropout-17           [-1, 16, 16, 16]               0
               Conv2d-18           [-1, 32, 16, 16]           4,608
                 ReLU-19           [-1, 32, 16, 16]               0
            GroupNorm-20           [-1, 32, 16, 16]              64
              Dropout-21           [-1, 32, 16, 16]               0
               Conv2d-22           [-1, 48, 16, 16]          13,824
                 ReLU-23           [-1, 48, 16, 16]               0
            GroupNorm-24           [-1, 48, 16, 16]              96
              Dropout-25           [-1, 48, 16, 16]               0
               Conv2d-26           [-1, 10, 16, 16]             480
                 ReLU-27           [-1, 10, 16, 16]               0
            GroupNorm-28           [-1, 10, 16, 16]              20
              Dropout-29           [-1, 10, 16, 16]               0
            MaxPool2d-30             [-1, 10, 8, 8]               0
               Conv2d-31             [-1, 16, 8, 8]           1,440
                 ReLU-32             [-1, 16, 8, 8]               0
            GroupNorm-33             [-1, 16, 8, 8]              32
              Dropout-34             [-1, 16, 8, 8]               0
               Conv2d-35             [-1, 32, 6, 6]           4,608
                 ReLU-36             [-1, 32, 6, 6]               0
            GroupNorm-37             [-1, 32, 6, 6]              64
              Dropout-38             [-1, 32, 6, 6]               0
               Conv2d-39             [-1, 64, 4, 4]          18,432
                 ReLU-40             [-1, 64, 4, 4]               0
            GroupNorm-41             [-1, 64, 4, 4]             128
              Dropout-42             [-1, 64, 4, 4]               0
            AvgPool2d-43             [-1, 64, 1, 1]               0
               Conv2d-44             [-1, 10, 1, 1]             640
    ================================================================
    Total params: 49,796
    Trainable params: 49,796
    Non-trainable params: 0
    ----------------------------------------------------------------
    Input size (MB): 0.01
    Forward/backward pass size (MB): 2.45
    Params size (MB): 0.19
    Estimated Total Size (MB): 2.65
    ----------------------------------------------------------------





    <function torchsummary.torchsummary.summary(model, input_size, batch_size=-1, device='cuda')>


### Performance:

parameters <50,000 and epoch <20, we have tracked the training and testing accuracies for all three models. The below mentioned table contains the best accuracy for training and test data after the 20th epoch: 

| Normalisation technique | Training accuracy  | Test Accuracy |
|-------------------------|--------------------|---------------|
| Batch Normalisation     |      69.30%        |    74.08%     |
| Layer Normalisation     |      61.12%        |    65.57%     |
| Group Normalisation     |      65.58%        |    69.89%     |

Note: The training accuracy is significantly lower than the test accuracy because the training data is augmented with multiple augmentation techniques which is expected to reduce the training accuracy. 

#### Training and test losses, accuracies for Batch Normalisation

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/843df3f2-39f8-44a6-820c-462c328162da)

#### Training and test losses, accuracies for Layer Normalisation

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/53b0b9b0-8767-4f56-baf4-81c46cc9a49f)

#### Training and test losses, accuracies for Group Normalisation

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/9830b4a0-468d-4315-9f88-f091f552b006)

#### Inference
Batch Normalisation performed better than both layer and group normalisation. The reason being the data within the channel carry similar information and the data across channels carry distinct information about the image. The channels are mostly independent of each other as they capture unique information about the image/previous channel. Hence batch normalisation worked better. 

### Misclassified image samples


#### Sample misclassified image for CNN with batch normalisation

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/5f3ff9e1-5928-4a80-8eda-dbdc03966c94)

#### Sample misclassified image for CNN with layer normalisation

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/ae66a665-1686-4207-90b4-1e3527f9aae1)

#### Sample misclassified image for CNN with group normalisation

![image](https://github.com/JVaishnavi/ERA_assgn8/assets/11015405/49d4df51-e5a1-4808-bbdb-69f2c9453565)

## Conclusion

Overall, for this dataset, batch normalisation works better than group normalisation and group normalisation works better than layer normalisation.

Batch Norm >= Group Norm >= Layer Norm for this dataset and the constraints

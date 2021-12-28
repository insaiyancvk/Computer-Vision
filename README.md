# Computer Vision: When your computer knows to see like you

### If you wanna check the notebooks, use the following (github gonna take forever to render them):


[Image Collection.ipynb](https://nbviewer.jupyter.org/github/insaiyancvk/Computer-Vision/blob/main/1.%20Image%20Collection.ipynb)

[Training and Detection.ipynb](https://nbviewer.jupyter.org/github/insaiyancvk/Computer-Vision/blob/main/2.%20Training%20and%20Detection.ipynb)

[Data Preprocess](https://nbviewer.jupyter.org/github/insaiyancvk/Computer-Vision/blob/main/Data%20Preprocess.ipynb)

Made my own perceptron on the iris dataset. [Check it out!](https://www.insaiyancvk.me/Computer-Vision/19CSE456/iris%20perceptron.html)

## Notes to self:

### Conda related stuff:

- Conda creating env in current working dir:

    ``` conda create --prefix ./<env name> ```

- Conda remove an env:
    
    ``` conda env remove -n <env name> ```

- Uninstall package in conda:
    
    ``` conda uninstall <package name> ```

- Clone a conda env:

    ``` conda create --prefix ./<env name> --copy --clone <env path> ```

- List all conda envs:

    ``` conda info --envs ```

- Increate the width of cells in ipynb:

```python
from IPython.core.display import display, HTML
display(HTML("<style>.container{width:100% !impotant;}</style>"))
```


### Non Conda stuff:

- Connect to camera with cv2:

```python
import cv2 # pip install opencv-python
cv2.VideoCapture(0) # 0 can be replaced with link to ipwebcam to use phone camera
```

### A general workflow of an end to end deep learning project on torch

#### If the image is in weird color (BGR):

```python
import cv2
RGB_img = cv2.cvtColor(w, cv2.COLOR_BGR2RGB)
```

#### Transform the images in training set:

```python
import torchvision.transforms as transforms
transformations = transforms.Compose([
    transforms.Resize([x,x]), # x can be any number (usually 224,224 or 256,256 is used)
    transforms.ToTensor(), # Converts the image array to Tensor
    transforms.Normalize(mean=[m,m,m],std=[s,s,s]) # "m" is the mean value that we are trying to achieve on each of RGB channels. "s" is the standard deviation from mean. (usually m=0.5 and s=0.5 are preferred)
])
```

#### Load train/test dataset, dataloader

```python
from torch.utils.data import DataLoader
import torchvision.datasets as datasets
<train/test>_dataset = datasets.ImageFolder(
    <path to train/test dir>,
    transform = transformations)
                                            
<train/test>_loader = DataLoader(
      dataset = <train/test>_dataset, 
      batch_size = <16/32>, 
      shuffle = <True/False>
) # always shuffle so that the model can generalise better
```

#### Print an image from <train/test> dataloader (transpose func)

```python
# This step has to be done after dataloader step is done

import matplotlib.pyplot as plt
import numpy as np

items = iter(<train/test>_loader)
image, label = items.next()
plt.imshow(np.transpose(image[n], (1,2,0)) # the transpose function rearranges the channels from [channels, height, width] to [height, width, channels]
```

#### Define the neural network

```python
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
    
        super(Net, self).__init__()
        
        # A pool function
        self.pool = nn.<pool_function>(n, n)
        
        # The number of layers have to be decided based on the image size
        
        # Conv layer 1
        self.conv1 = nn.Conv2d(<in_channels>, <out_channels>, <kernel_size>)
        # Conv layer 2
        self.conv2 = nn.Conv2d(<in_channels>, <out_channels>, <kernel_size>)
        # Conv layer 3
        self.conv3 = nn.Conv2d(<in_channels>, <out_channels>, <kernel_size>)
        # etc
        
        # Linear Layer 1
        self.fc1 = nn.Linear(<in_features>, <out_features>)
        # Linear Layer 2
        self.fc2 = nn.Linear(<in_features>, <out_features>)
        # Linear Layer 3
        self.fc3 = nn.Linear(<in_features>, <out_features>)
        # etc
    
    def forward(self, x):
        # x is images as Tensors after all the transformations
        
        # Conv layer 1 on x
        x = self.conv1(x)
        # Activation function on x
        x = F.<activation_function>(x)
        # Pool function on x
        x = self.pool(x)
        
        # Conv Layer 2 on x
        x = self.conv2(x)
        # Activation fucntion on x
        x = F.<activation_function>(x)
        # Pool function on x
        x = self.pool(x)
        
        # Conv Layer 3 on x
        x = self.conv3(x)
        # Activation fucntion on x
        x = F.<activation_function>(x)
        # Pool function on x
        x = self.pool(x)
        
        # etc
        
        # Flatten (placing all the rows in the tensor side by side)
        x = x.view(x.size(0), -1)
        
        # Linear layer 1 on x
        x = self.fc1(x)
        # Activation function on x
        x = F.<activation_function>(x)
        
        # Linear layer 2 on x
        x = self.fc2(x)
        # Activation function on x
        x = F.<activation_function>(x)
        
        # Linear layer 3 on x
        x = self.fc3(x)
        # Activation function on x
        x = F.<activation_function>(x) 
        
        # etc
        
        return x
```

#### Define each train step

**The steps:**
- Pass the image Tensors to the NN
- Calculate the loss
- Back propagation
- Update weights

```python
# This step as to be done after the Neural Network is defined

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

def train_step(model, images, labels, loss_func, optim, device): 
    
    # makes the gradiant 0 whenever called
    optim.zero_grad() 
    
    # Whatever the model predicts
    outputs = model(images.to(device)) # moving the image tensors to GPU/CPU (whichever "device" is available)
    
    # Calculate the gradiant
    loss = loss_func(outputs, labels.to(device))
    loss.backward()
    
    # Move towards the local minima
    optim.step()
    
    return loss.cpu().item() # move the loss value from GPU to CPU (.item() converts the tensor to int/float)

images, labels = iter(train_loader).items.next()
train_step(
        net, # net = Net() <model_object>
        images, 
        labels, 
        loss_func, # loss_func = nn.CrossEntropyLoss()  <loss_function>
        optim, # optim = optim.SGD(net.parameters(), lr = <some float number>, momentum = <some float number>) <optimizer>
        device = 'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
) 
```

- Resouruces
    - [Optimizer](https://pytorch.org/docs/stable/optim.html)
    - [Convolution functions, Pooling functions, Non-linear activation functions, Linear fuunctions, Loss Functions](https://pytorch.org/docs/stable/nn.functional.html)

#### Define an epoch

```python
# This step has to be done after train step is defined

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

def epoch(model, train_loader, loss_func, optim, device):
    
    mean_loss = 0
    
    # Move the model to GPU
    model = model.to(device)
    
    for image, label in train_loader:
        
        # Store the loss value returned from train_step
        loss = train_step(model, image, label, loss_func, optim, device)
        
        # Calculate mean loss
        mean_loss += loss/len(train_loader)
        
    return mean_loss
    
epoch(
    model_object,
    train_loader,
    criterion,
    optimizer,
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
)
```

### [Save and resume later](https://pytorch.org/tutorials/recipes/recipes/saving_and_loading_a_general_checkpoint.html)


### Other useful resources:

- Labelling images: [Label Img](https://github.com/tzutalin/labelImg).
- Some useful models in [TensorFlow](https://github.com/tensorflow/models/tree/master/research).
- Why [torchvision.transforms](https://stackoverflow.com/questions/50002543/what-are-transforms-in-pytorch-used-for) is used?
- Understand [loss and accuracy](https://datascience.stackexchange.com/questions/42599/what-is-the-relationship-between-the-accuracy-and-the-loss-in-deep-learning).
- [Gradients](https://www.javatpoint.com/gradient-with-pytorch) in pytorch.
- When to compute the [loss function](https://stats.stackexchange.com/questions/363885/is-a-loss-function-computed-after-each-step-of-gradient-descent-or-after-a-whole).
    - TL;DR for SGD calculate loss at the end of every epoch

### Deep learning related stuff:

- Types of architectures:

    - Single later feed forward network
    - Multilayer feed forward network
    - Single layer recurrent network
    - Multi layer recurrent network
    - Convolutional Neural Network
    - Long Short Term Memory (LSTM)

### Some insane models from ImageNet:

    1. LeNet-5
    2. AlexNet
    3. VGG-16
    4. Inception-V1
    5. Inception-V3
    6. ResNet-50
    7. Xception
    8. Inception-V4
    9. Inception-ResNet-V2
    10. ResNeXt-50


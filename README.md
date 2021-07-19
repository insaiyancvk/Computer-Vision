# Computer Vision: When your computer knows to see like you
## ~~A project under construction which ima resume during the semester~~

### If you wanna check the notebooks, use the following (github gonna take forever to render them):


[Image Collection.ipynb](https://nbviewer.jupyter.org/github/insaiyancvk/Computer-Vision/blob/main/1.%20Image%20Collection.ipynb)

[Training and Detection.ipynb](https://nbviewer.jupyter.org/github/insaiyancvk/Computer-Vision/blob/main/2.%20Training%20and%20Detection.ipynb)

[Data Preprocess](https://nbviewer.jupyter.org/github/insaiyancvk/Computer-Vision/blob/main/Data%20Preprocess.ipynb)

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

### Other useful resources:

- Labelling images: [Label Img](https://github.com/tzutalin/labelImg)
- Some useful models in [TensorFlow](https://github.com/tensorflow/models/tree/master/research)

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


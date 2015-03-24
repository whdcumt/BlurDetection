# BlurDetection
Blur Detection using Fast Fourier Transforms


## Quick Start
Getting the app to run is pretty easy. This script will not [install OpenCV](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html). However to install the rest of the project dependencies and run the demo script use the following commands.

```bash
# Clone the repo
git clone https://github.com/WillBrennan/SkinDetector && cd SkinDetector
# Install requirements
python setup.py install
# Run the Demonstration
python main.py <image_directory> --display --testing
```
## Usage
Usage of this as  a submodule is simple, just clone into your projects directory (or preferably add as a git submodule), and your ready to go. Below
is an example code usage.

```python
import os
import cv2
import numpy
import BlurDetector

img_path = raw_input("Please Enter Image Path")
assert os.path.exists(img_path), "img_path does not exsist"
image = cv2.imread(img_path)
val, blurry = BlurDetector.blur_detector(image)
print "this image {0} blurry".format(["isn't", "is"][blurry])
```

## References


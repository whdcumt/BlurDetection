# BlurDetection
Blur Detection using Fast Fourier Transforms. A Fast Fourier Transform is applied to the image using the default numpy functions, once this
is done the mean value in the transformed image is taken, this is then scaled with respect to the size of the image to compensate for the 
rippiling effect. This value is then used to threshold the image with larger values being indicative of an in focus image while lower values
of blurred images. This algorithm is also capable of generating masks of blurred images within the reason by using SLIC segmentation or the 
far faster `img_fft` thresholding and dialation operations. The second method mentioned is now default and can conduct blur detection in 
real time.




## Quick Start
Getting the app to run is pretty easy. This script will not [install OpenCV](http://docs.opencv.org/doc/tutorials/introduction/linux_install/linux_install.html). However to install the rest of the project dependencies and run the demo script use the following commands.

```bash
# Clone the repo
git clone https://github.com/WillBrennan/BlurDetection && cd BlurDetection
# Install requirements
sudo pip install numpy scikit-image
# Run the Demonstration
python main.py <image_directory> --display --mask
```
## Usage
Usage of this as  a submodule is simple, just clone into your projects directory (or preferably add as a git submodule), and your ready to go. Below is an example code usage. Under some circumstances manual tuning of the
threshold value may be required to achieve good results.

```python
import os
import cv2
import numpy
import BlurDetection

img_path = raw_input("Please Enter Image Path: ")
assert os.path.exists(img_path), "img_path does not exists"
img = cv2.imread(img_path)
img_fft, val, blurry = BlurDetection.blur_detector(img)
print "this image {0} blurry".format(["isn't", "is"][blurry])
msk, val = BlurDetection.blur_mask(img)
BlurDetection.scripts.display('img', img)
BlurDetection.scripts.display('msk', msk)
cv2.waitKey(0)
```

## References
[Image Partial Blur Detection and Classification](http://www.cse.cuhk.edu.hk/leojia/all_final_papers/blur_detect_cvpr08.pdf)

## Demonstration
![Blur Mask Demo](https://raw.githubusercontent.com/WillBrennan/BlurDetection/master/demo.png)

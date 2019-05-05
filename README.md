# CT metal object detection 
This repository contains the works for medical research scientist test!

## If the notebook failed to load in GitHub please try this:

[Test Notebook](https://nbviewer.jupyter.org/github/arcisad/Mirs/blob/master/mirs.ipynb)

You could find the lik to the challenge here:

[Medical Research Scientist Test](https://docs.google.com/document/d/1GwRRxhlzXWkB2XCnjOxApOcdWwkoPlZ4AJUzReO2E00/edit?usp=sharing)

## Table of contents:
- [Getting image information](#getting-image-information)
- [Segmenting metal objects](#segmenting-metal-objects)
- [U-Net image segmentation](#u-net-dnn-segmentation)

# Getting image information

Every DICOM scan image has a meta data built inside it, containing useful information about the image. Pixel spacing, slice thickness, patient position and orientation are the ones extracted for this section.

The class **dicom_object** generates the outputs for mean, max and centre of the scan. 

**Inputs** to the class object is the path to dicom images folder.

Method descriptions are below:

**Method**
```
get_meta
```

This method returns the dimensions, pixel spacing, slice spacing and slice thickness.


**Method**
```
get_max_intensity
```

Returns the maximum intensity od the scan. Hounsfield unit -2048 is removed to account for the scan margins. 

**Method**
```
get_mean_intensity
```

Returns the mean intensity od the scan. Hounsfield unit -2048 is removed to account for the scan margins. 

**Method**
```
get_center
```

Returns the center of the scan. Converts images ijk coordinates (pixel coordinates) to the native scanner coordinates via affine transformation. We read the patient position for the middle slice of the stack and carry out the affine transformation according to the following formula:

## Affine transformation

we have:

## xyz = A x ijk

where A is the affine transformation matrix. xyz is the scanner coordinates and ijk is the pixel coordinates in the image.

In matrix form for a single slice CT scanner:

![Affine](/images/render.png)

Where Ori is the patient orientation, Pos is the patient position in the slice, Ps is the pixel spacing. 

# Segmenting metal objects

The class **mask_object** segments the metal objects per slice and writes the corresponding binary mask files in the masks folder. 

CT images of the test are stores with Hounsfield units as intensities. Metal objects have much greater intensities than tissue and bone. Therefore, A simple thresholding followed by a morphology opening filter could generate acceptable result. Morphology opening is simply an erosion followed by dilation aiming for removing white spots outside of the segmented area. 

Link to the page: [Morphological opening](https://scikit-image.org/docs/dev/api/skimage.morphology.html#skimage.morphology.opening)


# U-Net DNN segmentation

The **unet_object** class is a Keras based U-net image segmentation method. [U-Net](https://lmb.informatik.uni-freiburg.de/people/ronneber/u-net/) developed at the University of Freiburg is a deep convolutional neural network architecture, especifically designed for medical image segmentation. It uses a encoder-decoder approach through a symmetrical pathway in which layers of the encoder are copied to the corresponding layers of the decoder. Downsampling the images to features and then upsampling them to the masks are normally used in segmentation DNNs such as Mask-RCNN or U-Net. 

A schematic view of U-net Architecture is seen below:


![U-Net Atchitecture](/images/u-net-architecture.png)

**Class methods are:**

**Method**
```
__init__
```

is the class constructor. Takes the following arguments:
- image_path: path to the DICOM images folder
- masks_path: path to the masks folder
- num_epochs: the number of epochs for trainig (default = 10)
- batch_size: size of the batch for feedng to the mode per step (default = 15)
- learning_rate: learning rate for the optimizer (default = 1e-4)
- dropout: dropout rate for the network (defualt = 0.5)
- iou_metric: IOU (intersection over union) metric for image similarity (optional, default = False)
- height: height of the image (default = 512)
- width: width of the image (default = 512)
- dim: image depth (dimensions), (default = 1 for a binary image)

## Assuming that the images should be fed to the network as is, I haven't done ant preprocessing. The only processing was removing -2048 intensity values fro the image. Contrast stretchig, cropping plus other image processing techniques coud improve the results.

The optimizer used is **Adam** optimizer and the loss function is a **weighted binary class entropy** function to handle the large class imbalance existing in the images. Dice loss function is a good option as well but should be used with class weights.


**Method**
```
unet_unit
```

Creates the model structure. Refer to the U-Net documentations for details about U-Net atcitechture.

I used data augmentation using the **Method** 

```
get_augmented_data
```

As the number of images (200) may not be enough for training a deep CNN, I used the convenient [ImageDataGenerator Class](https://keras.io/preprocessing/image/) from Keras to generate augmented data. It streams the generated images in batches when training the model.


# A metric for comparing real and generated masks.

There are many image similarity methods for comparing images, especially in object detection tasks. Euclidean distance is one of them and in binay images it is equivalent to sum squared error. Dice index is also a good option as it accounts for the intersection of the images.

**methods**
```
get_mse(mask_path, out_path)  ## inputs: paths to real masks (mask_path) and generated masks (out_path) folders
```
which returns values across all images,

and 
```
dice(mask, out)  ## inputs are single images. mask for the real and out for the generated mask.
```
which returns the dice index between two images.

calculates the metrics for image similarirty. 

# Pre-trained weights

## I have only run the network for 5 epochs. In reality it should run for many more, however, I had hardware and time limitations.

Pretrained weights for 5 epochs could be found here:

[Keras U-Net pre-trained weights for 5 epochs](https://drive.google.com/open?id=1klOjSrhEWq1eSJMJ445ud45ZV-oHjd_L)

# References

- [Defining the DICOM orientation](https://nipy.org/nibabel/dicom/dicom_orientation.html#dicom-slice-affine)
- [Coordinate systems](https://www.slicer.org/wiki/Coordinate_systems)
- (https://colab.research.google.com/drive/1BgCDxVdVc0MAe_kC0waMGUV9ShcWW0hM#scrollTo=hw66ln2H3YzQ)
- [unet for image segmentation](https://github.com/zhixuhao/unet/blob/master/model.py)
- [Yet another Keras U-net + data augmentation](https://www.kaggle.com/weiji14/yet-another-keras-u-net-data-augmentation)
- (https://gist.github.com/wassname/ce364fddfc8a025bfab4348cf5de852d)


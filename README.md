## SiamFC-PyTorch
This is the PyTorch (0.40) implementation of SiamFC tracker [1], which was originally <a href="https://github.com/bertinetto/siamese-fc">implemented</a> in Matlab using MatConvNet [2].

## Goal

* A more compact implementation of SiamFC [1]
* Reproduce the results of SiamFC [1], including data generation, training and tracking

## Requirements

* Python 2.7 (I use Anaconda 2.* here)
* Python-opencv
* PyTorch 0.40

## Data curation 

* Download <a href="http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz">ILSVRC15</a>, and unzip it (let's assume that `$ILSVRC2015_Root` is the path to your ILSVRC2015)
* Move `$ILSVRC2015_Root/Data/VID/val` into `$ILSVRC2015_Root/Data/VID/train/`, so we have five sub-folders in `$ILSVRC2015_Root/Data/VID/train/`
* It is a good idea to change the names of five sub-folders in `$ILSVRC2015_Root/Data/VID/train/` to `a`, `b`, `c`, `d`, and `e`
* Move `$ILSVRC2015_Root/Annotations/VID/val` into `$ILSVRC2015_Root/Annotations/VID/train/`, so we have five sub-folders in `$ILSVRC2015_Root/Annotations/VID/train/`
* Change the names of five sub-folders in `$ILSVRC2015_Root/Annotations/VID/train/` to `a`, `b`, `c`, `d` and `e`, respectively

* Generate image crops
  * Cd `$SiamFC-PyTorch/ILSVRC15-curation/` (Let's assume you've already downloaded the rep and its path is `$SiamFC-PyTorch`)

## Train

## Test (Tracking)

## Results

## References

[1] L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, and P. H. Torr. Fully-convolutional siamese networks for object tracking. In ECCV Workshop, 2016.

[2] A. Vedaldi and K. Lenc. Matconvnet – convolutional neural networks for matlab. In ACM MM, 2015.

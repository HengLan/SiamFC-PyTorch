## SiamFC-PyTorch
This is the PyTorch (0.40) implementation of SiamFC tracker [1], which was originally <a href="https://github.com/bertinetto/siamese-fc">implemented</a> in Matlab using MatConvNet [2]. In our implementation, we obtain better perforamnce than the original one.

## Goal

* A more compact implementation of SiamFC [1]
* Reproduce the results of SiamFC [1], including data generation, training and tracking

## Requirements

* Python 2.7 (I use Anaconda 2.* here)
* Python-opencv
* PyTorch 0.40
* other common packages such as `numpy`, etc

## Data curation 

* Download <a href="http://bvisionweb1.cs.unc.edu/ilsvrc2015/ILSVRC2015_VID.tar.gz">ILSVRC15</a>, and unzip it (let's assume that `$ILSVRC2015_Root` is the path to your ILSVRC2015)
* Move `$ILSVRC2015_Root/Data/VID/val` into `$ILSVRC2015_Root/Data/VID/train/`, so we have five sub-folders in `$ILSVRC2015_Root/Data/VID/train/`
* It is a good idea to change the names of five sub-folders in `$ILSVRC2015_Root/Data/VID/train/` to `a`, `b`, `c`, `d`, and `e`
* Move `$ILSVRC2015_Root/Annotations/VID/val` into `$ILSVRC2015_Root/Annotations/VID/train/`, so we have five sub-folders in `$ILSVRC2015_Root/Annotations/VID/train/`
* Change the names of five sub-folders in `$ILSVRC2015_Root/Annotations/VID/train/` to `a`, `b`, `c`, `d` and `e`, respectively

* Generate image crops
  * cd `$SiamFC-PyTorch/ILSVRC15-curation/` (Assume you've downloaded the rep and its path is `$SiamFC-PyTorch`)
  * change `vid_curated_path` in `gen_image_crops_VID.py` to save your crops
  * run `$python gen_image_crops_VID.py` (I run it in PyCharm), then you can check the cropped images in your saving path (i.e., `vid_curated_path`)
  
* Generate imdb for training and validation
  * cd `$SiamFC-PyTorch/ILSVRC15-curation/`
  * change `vid_root_path` and `vid_curated_path` to your custom path in `gen_imdb_VID.py`
  * run `$python gen_imdb_VID.py`, then you will get two json files `imdb_video_train.json` (~ 430MB) and `imdb_video_val.json` (~ 28MB) in current folder, which are used for training and validation

## Train

* cd `$SiamFC-PyTorch/Train/`
* Change `data_dir`, `train_imdb` and `val_imdb` to your custom <b>cropping path</b>, training and validation json files
* run `$python run_Train_SiamFC.py`
* <b>some notes in training</b>
  * the parameters for training are in `Config.py`
  * by default, I use GPU in training, and you can check the details in the function `train(data_dir, train_imdb, val_imdb, model_save_path="./model/", use_gpu=True)`
  * by default, the trained models will be saved to `$SiamFC-PyTorch/Train/model/`
  * each epoch (50 in total) may take 7-8 minuts (Nvidia 1080 GPU), and you can use parallelling utilities in PyTorch for speeding up
  * I tried to use fixed random seeds to get the same results, but it doesn't work ):, so results for each training may be slightly different (still better than the original)
  * <b>only</b> color images are used for training, and better performance is expected if using color+gray as in original paper

## Test (Tracking)

* cd `$SiamFC-PyTorch/Tracking/`
* <b>Firstly</b>, you should take a look at `Config.py`, which contains all parameters for tracking
* Change `self.net_base_path` to the path saving your trained models
* Change `self.seq_base_path` to the path storing your test sequences (OTB format, otherwise you need to revise the function `load_sequence()` in `Tracking_Utils.py`
* Change `self.net` to indicate whcih model you want for evaluation (by default, use the last one), and I've uploaded a trained model `SiamFC_50_model.pth` in this rep (located in $SiamFC-PyTorch/Train/model/)
* Change other parameters as your willing :)
* Now, let's run `$python run_Train_SiamFC.py`
* <b>some notes in tracking</b>
  * two evaluation types are provided: single video demo and evaluation on the whole (OTB-100) benchmark
  * you can also change whihc net for evaluation in `run_Train_SiamFC.py`

## Results
I tested the trained model on OTB-100 using a Nvidia 1080 GPU. The results and comparisons to the original implementation are shown in the below image. The running speed of our implementation is <b>82 fps</b>. Note that, both models are trained from stratch.

![image](/imgs/result.PNG)

## References

[1] L. Bertinetto, J. Valmadre, J. F. Henriques, A. Vedaldi, and P. H. Torr. Fully-convolutional siamese networks for object tracking. In ECCV Workshop, 2016.

[2] A. Vedaldi and K. Lenc. Matconvnet – convolutional neural networks for matlab. In ACM MM, 2015.

## Contact 

Any question are welcomed to hengfan@temple.edu.

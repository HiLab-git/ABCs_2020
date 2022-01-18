# The Second Place of MICCAI 2020 ABCs Challenge
[nnUNet_link]:https://github.com/MIC-DKFZ/nnUNetdescribe
[PyMIC_link]:https://github.com/HiLab-git/PyMIC
[ABCs_link]:https://abcs.mgh.harvard.edu/
This repository provides source code for MICCAI 2020 Anatomical brainBarriers to Cancer spread (ABCs) challenge. Method will be briefly introduced below, and our method won the 2nd place of [ABCs](ABCs_link).
Our method is based on [nnUNet][nnUNet_link], a self-adaptive segmentation method for medical images.

<img src='./HMRnet.png'  width="1100">

# Method overview
Our solution coatriaons corse and fine stage. We first use a localization model based on [nnUNet][nnUNet_link] to obtain a rough localization of five structures (Falx cerebri, Tentorium cerebelli, Sagittal & transverse brain sinuses, Cerebellum, Ventricles), and then use a High- and Multi-Resolution Network (HMRNet)  to  segment  each  structure  around  it's  local  region respectively, where a Bidirectional Feature Calibration (BFC) block  is  introduced  for  better  interaction  between  features  in the two branches. 

## Requirements
This code depends on [Pytorch](https://pytorch.org) (You need at least version 1.6), [PyMIC][PyMIC_link] and [nnUNet][nnUNet_link]. To use nnUNet, Download [nnUNet][nnUNet_link], and put them in the `ProjectDir` such as `/home/jk/project/ABCs`. 

## Usage
First, install the nnUNet and set environmental variables as follows.

```bash
cd nnUNet
pip install -e .
export nnUNet_raw_data_base="/home/jk/ABCS_data/nnUNet_raw_data_base"
export nnUNet_preprocessed="/home/jk/ABCS_data/nnUNet_preprocessed"
export RESULTS_FOLDER="/home/jk/ABCS_data/result"
```
# Coarse stage
### Data preparation
* Creat a path_dic for dataset ,  like`[ABCs_data_dir]="/home/jk/ABCS_data" `. Then dowload the dataset from [ABCs](ABCs_link) and put the dataset in the `ABCs_data_dir`, specifically, `ABCs_data_dir/nnUNet_raw_data/Task001_ABCs/imagesTr` for training images, `ABCs_data_dir/nnUNet_raw_data/Task001_ABCs/labelsTr` for training ground truth and `ABCs_data_dir/nnUNet_raw_data/Task001_ABCs/imagesTs` for test images. You can get more detailed guidance [here](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/dataset_conversion.md).

* Run the following commands to prepare training and testing data for nnUNet. Note that the input of nnUNet has three channels.`python  ./HNRNet/data_process/creat_data_json.py`
### Training
In the coarse stage, we only need the coarse location. In order to save unnecessary time, you can change `self.max_num_epochs = 1000` to `self.max_num_epochs = 45` in nnUNet/nnunet/training/network_training/nnUNetTrainerV2.py.
* Dataset conversion and preprocess. Run:
```bash
nnUNet_plan_and_preprocess -t 001 --verify_dataset_integrity
```
* Train 3D UNet. For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2  Task001_ABCs FOLD --npz
```
### Inference
Run the following command:
```bash
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t 001 -m 3d_fullres -f FOLD -chk model_best 
```

# Fine stage
First, you need to add the HMRNet into the nnUNet framework. 
* Add `HMRNet_p.py` and `HMRNet_s.py` into the `/nnUNet/nnunet/network_architecture/`.

* Replace the `nnunet/training/loss_functions/dice_loss.py` with the file `HMRNet/HMRNet/dice_loss.py`.

* Replace the `nnunet/training/network_training/nnUNetTrainerV2.py` with the file `HMRNet/HMRNet/nnUNetTrainerV2.py`.
## Data preparation and processing
* To separate each organ independently, run: 
```bash
python HMRNet/data_process/Extract_class_and_crop.py
```
* Especially, to get the  Sagittal brain sinu and Transverse brain sinus from Sagittal & transverse brain sinuses (organ3), run:
```bash
python HMRNet/data_process/Organ3_to_two_part.py
```

After extracting each organs, treated five types of organs as five tasks like `Task001_ABCs`, for example `Task002_ABCs_organ1,  Task003_ABCs_organ2 etc`. And put the corresponding into the bolders.
## Training
To segment each organs, you need trian six models.
For Cerebellum, Tentorium cerebelli and Ventricles, please set ` self.network = HMRNet_s(params) `  in  `nnunet/training/network_training/nnUNetTrainerV2.py`.
For Falx cerebri, Sagittal brain sinus and Transverse brain sinus, please set `self.network = HMRNet_p(params)` in `nnunet/training/network_training/nnUNetTrainerV2.py`.
In order to save unnecessary time, you can change self.max_num_epochs = 1000 to self.max_num_epochs = 400 in `nnUNet/nnunet/training/network_training/nnUNetTrainerV2.py`.
* Dataset conversion and preprocess. Run:
```bash
nnUNet_plan_and_preprocess -t TASK_name --verify_dataset_integrity
```
* Train 3D UNet. For FOLD in [0, 1, 2, 3, 4], run:
```bash
nnUNet_train 3d_fullres nnUNetTrainerV2  TASK_name FOLD --npz
```
`TASK_name` denotes the task_id of six sub-organs.

### Inference
To get the result of six model, run the following command respectively:
```bash
nnUNet_predict -i INPUT_FOLDER -o OUTPUT_FOLDER -t TASK_name -m 3d_fullres -f FOLD -chk model_best 
```
### Postprocessing
To merge   Sagittal brain sinu and Transverse brain sinus, run:
```bash
python HMRNet/data_process/Organ3_two_parts_fusion.py
```
To get the final segmentation result, you should pitch up the organ in succession. Run:
```bash
python HMRNet/data_process/Organs_fuse.py
```
(Attention:  you need to execute this command five times.)




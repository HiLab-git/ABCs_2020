import SimpleITK as sitk
import numpy as np
import os
from matplotlib import pyplot as plt 

#load_nii_as_narray
def load_nii_as_narray(filename):
    img_obj = sitk.ReadImage(filename)
    data_array = sitk.GetArrayFromImage(img_obj)
    return data_array

#save_arry_as_nii
def save_arry_as_nii(data,image_name,pixel_spacing=[1.2, 1.2, 1.2]):
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(pixel_spacing)
    sitk.WriteImage(img, image_name)
    
def save_array_as_volume(data, filename, transpose=True, pixel_spacing=[1, 1, 3]):
    """
    save a numpy array as nifty image
    inputs:
        data: a numpy array with shape [Channel, Depth, Height, Width]
        filename: the ouput file name
    outputs: None
    """
    if transpose:
        data = data.transpose(2, 1, 0)
    img = sitk.GetImageFromArray(data)
    img.SetSpacing(pixel_spacing)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(filename)
    writer.Execute(img)


#Extract sub-organ and cropping according the coarse segmentation 
coarseg_root ='/media/jk/data/data/eps_seg/valid'
coarseg_root ='/home/jk/ABCS_data/Organ_all/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ABCs/GT_val'
image_root ='/home/jk/ABCS_data/Organ_all/nnUNet_raw_data_base/nnUNet_raw_data/nnUNet_raw_data/Task001_ABCs/Va'
label_root = '/home/jk/shiyan/nnUnet_crop/fuse'
coarseg_save = '/home/jk/shiyan'
image_save = '/media/jk/data/data/single_crop_by_gt/traindata'
label_save = '/home/jk/shiyan/nnUnet_crop'
label_wanted = [1,2,3,4,5]                                                                       
pading = [10,10,10]


crop_coarsegs = False
crop_images = False
crop_labels = True


for file in os.listdir(coarseg_root):  #  the file end with  "_0006.nii.gz"
    sub_coarseg = os.path.join(coarseg_root,file)
    sub_coarseg = load_nii_as_narray(sub_coarseg)
    
    
    sub_coarseg_size = sub_coarseg.shape
    
#Extract  label  wanted 
    for i in range(len(label_wanted)): 
        
        ncoarseg = np.zeros_like(sub_coarseg)
        ncoarseg[np.where(sub_coarseg==label_wanted[i])]= 1
        sub_coarseg_2 = ncoarseg
#get the coordinate 
        nonzeropoint = np.asarray(np.nonzero(sub_coarseg_2))

        maxpoint = np.max(nonzeropoint,1).tolist()

        minpoint = np.min(nonzeropoint,1).tolist()

        for ii in range(len(pading)):
            maxpoint[ii] = min(maxpoint[ii]+pading[ii],sub_coarseg_size[ii])#sub_coarseg_size
            minpoint[ii] = max(minpoint[ii]-pading[ii],0)


#cropping  coarseg
        if crop_coarsegs :
            sub_coarseg_3 = sub_coarseg_2[minpoint[0]:maxpoint[0], 
                                  minpoint[1]:maxpoint[1], 
                                  minpoint[2]:maxpoint[2]]
            print('coarseg:',sub_coarseg_3.shape)
            coarseg_name = os.path.join(coarseg_save,str(i+1),file)
            save_array_as_volume(sub_coarseg_3,coarseg_name,transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
            print(coarseg_name)
        
#croppting label
        if crop_labels:
            file_c = file.replace('_0006.nii.gz','.nii.gz')
            sub_label = os.path.join(label_root,file_c)
            sub_label = load_nii_as_narray(sub_label)
        
            nsub_label = np.zeros_like(sub_label)
            nsub_label[np.where(sub_label==label_wanted[i])]= 1
        
            sub_label_2 = nsub_label[minpoint[0]:maxpoint[0], 
                              minpoint[1]:maxpoint[1], 
                              minpoint[2]:maxpoint[2]]
            print('label:',sub_label_2.shape)
            label_name = os.path.join(label_save,str(i+1),file_c)
            save_array_as_volume(sub_label_2,label_name,transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
            print(label_name)
        
#crop ct t1 t2   data set
        if crop_images:
            ct = file.replace('_0006.nii.gz', '_0000.nii.gz')
            t1 = file.replace('_0006.nii.gz', '_0001.nii.gz')
            t2 = file.replace('_0006.nii.gz', '_0002.nii.gz')
            modes = [ct,t1,t2]
            for mode in modes:
                sub_image = os.path.join(image_root,mode)
                sub_image = load_nii_as_narray(sub_image)
                sub_image_1 = sub_image[minpoint[0]:maxpoint[0], 
                                  minpoint[1]:maxpoint[1], 
                                  minpoint[2]:maxpoint[2]]
                print('image:',sub_image_1.shape)
                image_name = os.path.join(image_save,str(i+1),mode)
                save_array_as_volume(sub_image_1,image_name,transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
                print(image_name)

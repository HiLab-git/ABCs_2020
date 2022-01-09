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

# Split the organ 3 into two parts
data_root = '/home/jk/ABCS_data/Organ_all/nnUNet_raw_data_base/nnUNet_raw_data/nnUNet_raw_data/Task001_ABCs/labelsTr'
part1_save_root ='/home/jk/organ3_crop/organ3_part1/gt'
part1_save_root = '/home/jk/organ3_crop/organ3_part2/gt'
image_root = '/home/jk/ABCS_data/Organ_all/nnUNet_raw_data_base/nnUNet_raw_data/nnUNet_raw_data/Task001_ABCs/imagesTr'
part1_image_save = '/home/jk/organ3_crop/organ3_part1/data'
part2_image_save = '/home/jk/organ3_crop/organ3_part2/data'
crop_first_part=True
crop_second_part = True
pading = [5,5,5]

# part one
for file in os.listdir(data_root):
    data_path = os.path.join(data_root,file)
    data = load_nii_as_narray(data_path)3
    median = int(data.shpe[2]/2)
    
    ndata = np.zeros_like(data)
    ndata[np.where(data==3)= 1
    ndata_part1 = ndata[...,median-20:median+20]
    ndata_part1_shape = ndata_part1.shape


    nonzeropoint = np.asarray(np.nonzero(ndata_part1))
    maxpoint = np.max(nonzeropoint,1).tolist()
    minpoint = np.min(nonzeropoint,1).tolist()

    for ii in range(len(pading)):
        maxpoint[ii] = min(maxpoint[ii]+pading[ii],sub_coarseg_size[ii])
        minpoint[ii] = max(minpoint[ii]-pading[ii],0)
        a[ii] = maxpoint[ii]-minpoint[ii]    

    median = int(data.shape[2]/2)
    data = data[...,median-20:median+20]
    name = os.path.join(part1_save_root,file)
    save_array_as_volume(data, name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
    print(name)
    
    if crop_images:
        ct = file.replace('.nii.gz', '_0000.nii.gz')
        t1 = file.replace('.nii.gz', '_0001.nii.gz')
        t2 = file.replace('.nii.gz', '_0002.nii.gz')
        modes = [ct,t1,t2]
        for mode in modes:
            sub_image = os.path.join(image_root,mode)
            sub_image = load_nii_as_narray(sub_image)
            sub_image_1 = sub_image[...,median-20:median+20]
            image_name = os.path.join(part1_image_save,str(i+1),mode)
            save_array_as_volume(sub_image_1,image_name,transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
            print(image_name)
print('part 1,  done!!!!!')



#part two
for file in os.listdir(data_root):
    data_path = os.path.join(data_root,file)
    data = load_nii_as_narray(data_path)
    data = data[:50]
    name = os.path.join(part2_save_root,file)
    save_array_as_volume(data, name, transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
    
    if crop_images:
        ct = file.replace('.nii.gz', '_0000.nii.gz')
        t1 = file.replace('.nii.gz', '_0001.nii.gz')
        t2 = file.replace('.nii.gz', '_0002.nii.gz')
        modes = [ct,t1,t2]
        for mode in modes:
            sub_image = os.path.join(image_root,mode)
            sub_image = load_nii_as_narray(sub_image)
            sub_image_1 = sub_image[:50]
            image_name = os.path.join(part2_image_save,mode)
            save_array_as_volume(sub_image_1,image_name,transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
            print(image_name)
print('part 2, done!!!!!')






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

# combine single organ to get  the final result
coarseg_root ='/media/jk/data/data/eps_seg/valid'
organ_root ='/home/jk/shiyan/addation/BFC_top/5'
save = '/home/jk/shiyan/addation/BFC_top/fuse'
organ_wanted=5
pading = [10,10,10]


for file in os.listdir(coarseg_root):
    sub_coarseg = os.path.join(coarseg_root,file)# coarseg path
    sub_coarseg = load_nii_as_narray(sub_coarseg)
    sub_coarseg_size = sub_coarseg.shape
    
    sub_coarseg2 = os.path.join(save,file)
    sub_coarseg2 = load_nii_as_narray(sub_coarseg2)

#Get the cropped coordinates
    location = np.where(sub_coarseg==organ_wanted)

    maxpoint = np.max(location, 1).tolist()

    minpoint = np.min(location, 1).tolist()

    for i in range(len(pading)):
        maxpoint[i] = min(maxpoint[i]+pading[i],sub_coarseg_size[i])#sub_coarseg_size
        minpoint[i] = max(minpoint[i]-pading[i],0)
        
# Exract the  volume， according  to the  coordinates
    organ_coarseg = os.path.join(organ_root,file)
    organ_coarseg = load_nii_as_narray(organ_coarseg)
    

    block =sub_coarseg2[minpoint[0]:maxpoint[0], 
                        minpoint[1]:maxpoint[1], 
                        minpoint[2]:maxpoint[2]]
    
    block[np.where(block==organ_wanted)]=0    
    block[np.where(organ_coarseg==1)]=organ_wanted
    sub_coarseg2[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]]=block
    
    name = os.path.join(save,file)
    save_array_as_volume(sub_coarseg2,name,transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
print("done")

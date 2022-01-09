#ABCs organ3 two parts fusion
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



data_root ='/media/jk/data/融合/原始gt'
part1 = '/media/jk/data/融合/fuse_loss/loss1_1.5/part1'
part2_root ='/media/jk/data/融合/fuse_loss/loss1_1.5/part2'
save = '/media/jk/data/融合/fuse_loss/loss1_1.5/fuse'
pading = [5,5,5]




for file in os.listdir(data_root):
    data_path = os.path.join(data_root,file)
    part_path = os.path.join(part1,file)
    part = load_nii_as_narray(part_path)
    part_path2 = os.path.join(part2_root,file)
    part2 = load_nii_as_narray(part_path2)    
    data = load_nii_as_narray(data_path)
    data_shape = data.shape
    median = int(data_shape[2]/2)
    
    ndata = np.zeros_like(data) 
    Adata = np.zeros_like(data)*1
    AAdata = np.zeros_like(data)*1
    ndata[np.where(data==3)]= 1
    
    ndata_part1 = ndata[...,median-20:median+20]
    ndata_part1_zero = np.zeros_like(ndata_part1)
    ndata_part1_shape = ndata_part1.shape
    
    ndata_part2 = ndata[0:55]
    ndata_part2_zero = np.zeros_like(ndata_part2)
    ndata_part2_shape = ndata_part2.shape
#part2获取坐标点
    nonzeropoint2 = np.asarray(np.nonzero(ndata_part2))
#         print(nonzeropoint)
    maxpoint2 = np.max(nonzeropoint2,1).tolist()
#         print(maxpoint)
    minpoint2 = np.min(nonzeropoint2,1).tolist()
#         a=np.zeros(3)    
    for ii in range(len(pading)):
        maxpoint2[ii] = min(maxpoint2[ii]+pading[ii],ndata_part2_shape[ii])
        minpoint2[ii] = max(minpoint2[ii]-pading[ii],0)      
    
    block2 = part2
    ndata_part2_zero[minpoint2[0]:maxpoint2[0], minpoint2[1]:maxpoint2[1], minpoint2[2]:maxpoint2[2]] = block2
    print(file,block2.shape)
    
    Adata[0:55] = ndata_part2_zero   
    AAdata = Adata*1

#part1获取坐标点
    nonzeropoint = np.asarray(np.nonzero(ndata_part1))
#         print(nonzeropoint)
    maxpoint = np.max(nonzeropoint,1).tolist()
#         print(maxpoint)
    minpoint = np.min(nonzeropoint,1).tolist()
#         a=np.zeros(3)    
    for ii in range(len(pading)):
        maxpoint[ii] = min(maxpoint[ii]+pading[ii],ndata_part1_shape[ii])
        minpoint[ii] = max(minpoint[ii]-pading[ii],0)    
   
    block = part
#     print(file,block.shape)
    ndata_part1_zero[minpoint[0]:maxpoint[0], minpoint[1]:maxpoint[1], minpoint[2]:maxpoint[2]] = block
    block3 =AAdata[...,median-20:median+20]
    Adata[...,median-20:median+20] = block3 +ndata_part1_zero
    Adata[np.where(Adata==2)]= 1
    

    
    name = os.path.join(save,file)
    save_array_as_volume(Adata,name,transpose=False, pixel_spacing=[1.2, 1.2, 1.2])
    print(file)
    print(name)
print("done")
    
from collections import OrderedDict
import SimpleITK as sitk
from batchgenerators.utilities.file_and_folder_operations import *

if __name__ == "__main__":
    folder = "/home/jk/ABCS_data/no_con/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ABCs/labelsTr"
    out_folder = "/home/jk/ABCS_data/no_con/nnUNet_raw_data_base/nnUNet_raw_data/Task001_ABCs/labelsTr"
    raw_data =  os.listdir(folder)
    json_dict = OrderedDict()
    json_dict['name'] = "ABCs2020"
    json_dict['description'] = "ABCs"
    json_dict['tensorImageSize'] = "4D"
    json_dict['reference'] = "see challenge website"
    json_dict['licence'] = "see challenge website"
    json_dict['release'] = "0.0"
    json_dict['modality'] = {
        "0": "CT",
        "1":"T1",
        "2":"T2",

    }
    json_dict['labels'] = {
        "0": "background",
        "1": "Cerebellum",
        "2": "Falx cerebri",
        "3": "Sagittal & transverse brain sinuses",
        "4": "Tentorium cerebelli",
        "5": "Ventricles"
    }
    json_dict['numTraining'] = len(raw_data)
    json_dict['numTest'] = 0
    json_dict['training'] = [{'image': "./imagesTr/%s" % i, "label": "./labelsTr/%s" % i} for i in raw_data]
    json_dict['test'] =[]

    save_json(json_dict, os.path.join(out_folder, "dataset.json"))
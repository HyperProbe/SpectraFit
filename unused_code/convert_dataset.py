import os

import torch
from spectral import open_image


def BilFile_to_Tensor(img):
    np_img = img.read_subimage(range(img.shape[0]), range(img.shape[1])).astype(float)
    tensor_img = torch.from_numpy(np_img).float()
    tensor_img = tensor_img.permute(2, 0, 1)
    return tensor_img


def convert_hdr():
    path = "dataset" + "/HSI_Human_Brain_Database_IEEE_Access/"
    patient_dirs = os.listdir(path)
    patient_dirs = [directory for directory in patient_dirs if os.path.isdir(os.path.join(path, directory))]
    print(patient_dirs)
    for dir in patient_dirs:
        if not os.path.exists("{}/raw.pt".format(path + dir)):
            hdr_path = "{}/raw.hdr".format(path + dir)
            print(hdr_path)
            img = BilFile_to_Tensor(open_image(hdr_path))
            torch.save(img, "{}/raw.pt".format(path + dir))
            print("saved raw")
        if not os.path.exists("{}/gtMap.pt".format(path + dir)):
            gt_path = "{}/gtMap.hdr".format(path + dir)
            print(gt_path)
            gt = BilFile_to_Tensor(open_image(gt_path)).int()
            torch.save(gt, "{}/gtMap.pt".format(path + dir))
            print("saved gtMap")
        if not os.path.exists("{}/darkReference.pt".format(path + dir)):
            dark_path = "{}/darkReference.hdr".format(path + dir)
            print(dark_path)
            gt = BilFile_to_Tensor(open_image(dark_path)).int()
            torch.save(gt, "{}/darkReference.pt".format(path + dir))
            print("saved darkReference")
        if not os.path.exists("{}/whiteReference.pt".format(path + dir)):
            white_path = "{}/whiteReference.hdr".format(path + dir)
            print(white_path)
            gt = BilFile_to_Tensor(open_image(white_path)).int()
            torch.save(gt, "{}/whiteReference.pt".format(path + dir))
            print("saved whiteReference")


convert_hdr()

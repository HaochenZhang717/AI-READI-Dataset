import pydicom
import numpy as np
import cv2
from tqdm import tqdm
import os


def read_dcm_image(path):
    """
    Read a DICOM (.dcm) file and return the pixel image as a numpy array.

    Args:
        path (str): path to the .dcm file

    Returns:
        image (np.ndarray): image array
        ds (pydicom.Dataset): dicom metadata
    """

    ds = pydicom.dcmread(path)

    # image pixel data
    img = ds.pixel_array.astype(np.float32)

    # normalize if needed (helps for visualization)
    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    return img, ds


def dcm_to_png(dcm_path, png_path):
    """
    Convert a DICOM image to PNG.
    """

    ds = pydicom.dcmread(dcm_path)

    img = ds.pixel_array.astype(np.float32)

    # normalize to 0-255
    img -= img.min()
    if img.max() > 0:
        img /= img.max()

    img = (img * 255).astype(np.uint8)

    # if grayscale convert to 3-channel
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

    cv2.imwrite(png_path, img)


def convert_dataset(root_dir, save_dir):
    """
    Convert all DICOM files under root_dir to PNG.
    """

    os.makedirs(save_dir, exist_ok=True)

    patient_ids = sorted(os.listdir(root_dir))

    for pid in tqdm(patient_ids):

        patient_dir = os.path.join(root_dir, pid)

        if not os.path.isdir(patient_dir):
            continue

        save_patient_dir = os.path.join(save_dir, pid)
        os.makedirs(save_patient_dir, exist_ok=True)

        for file in os.listdir(patient_dir):

            if not file.endswith(".dcm"):
                continue

            dcm_path = os.path.join(patient_dir, file)

            png_name = file.replace(".dcm", ".png")
            png_path = os.path.join(save_patient_dir, png_name)

            dcm_to_png(dcm_path, png_path)


if __name__ == "__main__":
    # img, meta = read_dcm_image(
    #     "/Users/zhc/Downloads/1001_maestro2_3d_macula_oct_cfp_l_2.16.840.1.114517.10.5.1.4.907063120230727165807.2.1.dcm")
    # # retinal_photography/cfp/topcon_maestro2/
    # blood_glucose_file_path = f"/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/retinal_photography/cfp/topcon_maestro2/"
    #
    # print(img.shape)
    #
    # plt.imshow(img, cmap="gray")
    # plt.axis("off")
    # plt.show()

    root_dir = "/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/retinal_photography/cfp/topcon_maestro2/"
    save_dir = "/playpen/haochenz/retinal_photography/cfp/topcon_maestro2/"

    convert_dataset(root_dir, save_dir)
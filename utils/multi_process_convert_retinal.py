import pydicom
import numpy as np
import cv2
from tqdm import tqdm
import os
from multiprocessing import Pool, cpu_count




# def dcm_to_png_worker(args):
#
#     dcm_path, png_path = args
#
#     try:
#         ds = pydicom.dcmread(dcm_path)
#
#         img = ds.pixel_array
#
#         # 如果是 grayscale
#         if img.ndim == 2:
#             img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
#
#         # 如果是 RGB，需要转 BGR 才能被 cv2 正确保存
#         elif img.shape[2] == 3:
#             img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
#
#         cv2.imwrite(png_path, img, [cv2.IMWRITE_PNG_COMPRESSION, 3])
#
#     except Exception as e:
#         print(f"Failed: {dcm_path} | {e}")


def dcm_to_png_worker(args):

    dcm_path, png_path = args

    try:
        ds = pydicom.dcmread(dcm_path)

        img = ds.pixel_array

        # grayscale → RGB
        if img.ndim == 2:
            img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        # RGB → BGR (cv2)
        elif img.shape[2] == 3:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        # ===== downsample 2x =====
        h, w = img.shape[:2]
        img = cv2.resize(
            img,
            (w // 2, h // 2),
            interpolation=cv2.INTER_AREA
        )

        cv2.imwrite(
            png_path,
            img,
            [cv2.IMWRITE_PNG_COMPRESSION, 3]
        )

    except Exception as e:
        print(f"Failed: {dcm_path} | {e}")


def convert_dataset_parallel(root_dir, save_dir, num_workers=None):
    """
    Parallel version of dataset conversion
    """

    if num_workers is None:
        num_workers = cpu_count()

    print(f"Using {num_workers} workers")

    os.makedirs(save_dir, exist_ok=True)

    tasks = []

    patient_ids = sorted(os.listdir(root_dir))

    for pid in patient_ids:

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

            tasks.append((dcm_path, png_path))

    print(f"Total files: {len(tasks)}")

    with Pool(num_workers) as pool:
        list(tqdm(pool.imap_unordered(dcm_to_png_worker, tasks), total=len(tasks)))


if __name__ == "__main__":

    root_dir = "/playpen-shared/mshuang/morris/morris/d9ef6cf1-f6c3-4956-a91e-adf409e105f0/dataset/retinal_photography/cfp/topcon_maestro2/"
    save_dir = "/playpen/haochenz/AI-READI/retinal_photography/cfp/topcon_maestro2/"

    convert_dataset_parallel(root_dir, save_dir)
import numpy as np
import os
from IPython.display import clear_output
from PIL import Image
from multiprocessing import Pool, freeze_support, cpu_count
import os

def of_score_calc(u_img, v_img, method="mean"):
    u = np.array(Image.open(u_img))
    v = np.array(Image.open(v_img))
    magnitude = np.sqrt(u ** 2 + v ** 2)
    if method == "90th_percentile":
        return np.percentile(magnitude, 90)
    if method == "mean":
        return np.mean(magnitude)

def wrapped_of_score_calc(args):
    return of_score_calc(*args)

def make_score_list(root_folder, result_folder, method="mean"):
    pool = Pool(cpu_count())
    u_path = os.path.join(root_folder, "u/")
    v_path = os.path.join(root_folder, "v/")
    subfolders_u = [f.path for f in os.scandir(u_path) if f.is_dir()]
    subfolders_v = [f.path for f in os.scandir(v_path) if f.is_dir()]
    already_created = [f.path for f in os.scandir(result_folder) if f.is_file()]
    if len(subfolders_u) != len(subfolders_v):
        raise ValueError("Not the same number of folders in u/ and v/")
    for s, sub_u in enumerate(subfolders_u):
        clear_output()
        print("subfolder = {} ({}/{})".format(sub_u, s+1, len(subfolders_u)))
        dir_name = sub_u.split("/")[-1]
        res_path = os.path.join(result_folder, dir_name)
        if not(res_path in already_created):
            sub_v = subfolders_v[s]
            u_images = [f.path for f in os.scandir(sub_u) if f.is_file()]
            v_images = [f.path for f in os.scandir(sub_v) if f.is_file()]
            img_list = [(u_images[i], v_images[i]) for i in range(len(u_images))]
            score_list = pool.map(wrapped_of_score_calc, img_list)
            np.savetxt(res_path, score_list, fmt="%.4f")
        else:
            print("skipping")

def moving_average(a, n):
    res = np.zeros(len(a) - n + 1)
    res[0] = np.mean(a[:n])
    for i in range(1, len(res)):
        res[i] = res[i-1]+(a[i+n-1]-a[i-1])/n
    return res
    
def of_slicing(of_folder, file_name, width = 100):
    path = os.path.join(of_folder, file_name)
    score_ary = np.loadtxt(path)
    if len(score_ary)<=width:
        return 0
    else:
        ma = moving_average(score_ary, width)
        return np.argmax(ma)
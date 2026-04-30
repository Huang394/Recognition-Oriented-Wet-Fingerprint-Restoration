import numpy as np
import glob
import os
import cv2
import shutil
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.neighbors import KNeighborsClassifier

from preprocess import find_large_background_area
from wet_quality_metrices import grayscale_mean, grayscale_std, black_pixel_ratio, otsu_black_ratio, sobel_gradient_magnitude, block_info
from quality_votes import bad_quality_vote

plt.switch_backend('agg')

img_folder = fr"C:\Users\user\Desktop\Coding\Project\fingerprint\Dataset\Nasic9395_testset\test\hazy"
knn_trainset_folder = fr"./knn_dataset"
result_folder = fr"./Nasic9395_testset\identified"
metrices_path = ['gray_mean', 'gray_std', 'black_pixel_ratio', 'otsu_black_ratio', 'gradient_magnitude', 'lap_spectrum']
quality_metrices_num = 6
quality_levels = 3

def compute_wet_quality(img_root, img_name_list):
    gray_mean_arr = []
    gray_std_arr = []
    black_pixel_ratio_arr = []
    otsu_black_ratio_arr = []
    gradient_magnitude_arr = []
    lap_spectrum_arr = []

    for img_fname in tqdm(img_name_list):
        img_path = os.path.join(img_root, img_fname)
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray_slide_img = gray_img[:,2:-2]

        background_img, large_background_loc, white_value = find_large_background_area(gray_slide_img)

        gray_mean = grayscale_mean(gray_slide_img, background_img)
        gray_std = grayscale_std(gray_slide_img, background_img)
        black_pixel_r = black_pixel_ratio(gray_img, background_img)
        otsu_black_r, _, _ = otsu_black_ratio(gray_slide_img, white_value)
        gradient_m = sobel_gradient_magnitude(gray_slide_img)
        lap_pos, power_spectrum_2d_norm = block_info(gray_slide_img)

        gray_mean_arr.append(gray_mean)
        gray_std_arr.append(gray_std)
        black_pixel_ratio_arr.append(black_pixel_r)
        otsu_black_ratio_arr.append(otsu_black_r)
        gradient_magnitude_arr.append(gradient_m)
        lap_spectrum_arr.append(power_spectrum_2d_norm)

    return gray_mean_arr, gray_std_arr, black_pixel_ratio_arr, otsu_black_ratio_arr, gradient_magnitude_arr, lap_spectrum_arr

def find_standard_score_interval(metrices_array, std_mode = True):
    mean = np.mean(metrices_array)
    std = np.std(metrices_array)
    std_score_arr = [0, mean-2*std, mean-std, mean, mean+std, mean+2*std, np.inf]

    percent_score_arr = [np.percentile(metrices_array, i*10) for i in range(11)]
    if std_mode:
        return std_score_arr
    else:
        return percent_score_arr
    
def quality_score_evaluate(wet_q_values, wet_q_scores):
    wet_level_arr = [0 for i in range(quality_metrices_num)]
    for m_id in range(quality_metrices_num):
            wet_q = wet_q_values[m_id]
            wet_q_score = wet_q_scores[m_id]
            for score_id in range(len(wet_q_score) - 1):
                if (wet_q>=wet_q_score[score_id]) and (wet_q<wet_q_score[score_id+1]):
                    wet_level_arr[m_id] = score_id
    return wet_level_arr

if __name__ == "__main__":
    ## remove bad quality fingerprints
    all_img_name_list = os.listdir(img_folder)
    gray_mean_arr, gray_std_arr, black_pixel_ratio_arr, otsu_black_ratio_arr, gradient_magnitude_arr, lap_spectrum_arr = compute_wet_quality(img_folder, all_img_name_list)
    
    gray_mean_score_arr = find_standard_score_interval(gray_mean_arr)
    gray_std_score_arr = find_standard_score_interval(gray_std_arr)
    black_pixel_ratio_score_arr = find_standard_score_interval(black_pixel_ratio_arr)
    otsu_black_ratio_score_arr = find_standard_score_interval(otsu_black_ratio_arr)
    gradient_magnitude_score_arr = find_standard_score_interval(gradient_magnitude_arr)
    lap_spectrum_score_arr = find_standard_score_interval(lap_spectrum_arr)
    wet_q_scores = [gray_mean_score_arr, gray_std_score_arr, black_pixel_ratio_score_arr, otsu_black_ratio_score_arr, gradient_magnitude_score_arr, lap_spectrum_score_arr]
    
    os.makedirs(os.path.join(result_folder, str(quality_levels)), exist_ok=True)
    bad_vote_counts_arrs = [0 for i in range(quality_metrices_num+2)]
    bad_img_name_list = []
    for id, img_fname in tqdm(enumerate(all_img_name_list)):
        img_path = os.path.join(img_folder, img_fname)
        gray_mean = gray_mean_arr[id]
        gray_std = gray_std_arr[id]
        black_pixel_r = black_pixel_ratio_arr[id]
        otsu_black_r = otsu_black_ratio_arr[id]
        gradient_m = gradient_magnitude_arr[id]
        power_spectrum_2d_norm = lap_spectrum_arr[id]
        
        wet_q_values = [gray_mean, gray_std, black_pixel_r, otsu_black_r, gradient_m, power_spectrum_2d_norm]

        wet_level_arr = quality_score_evaluate(wet_q_values, wet_q_scores)
        bad_result, bad_vote = bad_quality_vote(wet_level_arr, bad_vote_thresh = 2)
        bad_vote_counts_arrs[bad_vote] += 1
        
        if bad_result:
            bad_img_name_list.append(img_fname)
            shutil.copyfile(img_path, os.path.join(result_folder, str(quality_levels), img_fname))
    ## ##
    
    
    for i in range(quality_levels):
        if os.path.exists(os.path.join(result_folder, str(i))):
            shutil.rmtree(os.path.join(result_folder, str(i)))
        os.makedirs(os.path.join(result_folder, str(i)), exist_ok=True)
    
    ## Build KNN ##
    print('Build Classification Model')
    print('-'*80)
    # load data
    knn_x_train = np.load(os.path.join(knn_trainset_folder, 'x_train_q5.npy'))
    knn_y_train = np.load(os.path.join(knn_trainset_folder, 'y_train_q5.npy'))
    print("x_train shape: ", knn_x_train.shape)
    print("y_train shape: ", knn_y_train.shape)

    # build knn
    KNN = KNeighborsClassifier(n_neighbors=9)
    KNN.fit(knn_x_train, knn_y_train)
    print('-'*80)
    ## ##
    
    bad_img_name_list = set(bad_img_name_list)
    good_img_name_list = [i for i in all_img_name_list if i not in bad_img_name_list]

    inference_wet_count_arr = [0 for i in range(quality_levels)]
    for img_fname in tqdm(good_img_name_list):
        img_path = os.path.join(img_folder, img_fname)
        gray_img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        gray_slide_img = gray_img[:,2:-2]

        background_img, large_background_loc, white_value = find_large_background_area(gray_slide_img)

        gray_mean = grayscale_mean(gray_slide_img, background_img)
        gray_std = grayscale_std(gray_slide_img, background_img)
        black_pixel_r = black_pixel_ratio(gray_img, background_img)
        otsu_black_r, _, _ = otsu_black_ratio(gray_slide_img, white_value)
        gradient_m = sobel_gradient_magnitude(gray_slide_img)
        lap_pos, power_spectrum_2d_norm = block_info(gray_slide_img)
        
        wet_q_values = np.array([gray_std, black_pixel_r, otsu_black_r, gradient_m, power_spectrum_2d_norm])
        wet_q_values = np.reshape(wet_q_values, (1, -1))
        pred = KNN.predict(wet_q_values)
        level = pred[0]
        inference_wet_count_arr[level] += 1

        shutil.copyfile(img_path, os.path.join(result_folder, str(level), img_fname))
        
    print(inference_wet_count_arr)
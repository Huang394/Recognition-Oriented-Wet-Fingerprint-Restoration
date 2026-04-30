import os
import glob
import shutil

if __name__ == '__main__':
    image_folder = fr"C:\Users\user\Desktop\Coding\Project\fingerprint\Dataset\Nasic9395_testset\test\Nasic9395_testset_test_arcface\wet\unclassified"
    sorted_folder = fr"./Nasic9395_testset\identified"
    result_folder = fr"./Nasic9395_testset\wet"
    quality_levels = 3

    for i in range(quality_levels + 1):
        result_path = os.path.join(result_folder, str(i))
        os.makedirs(result_path,exist_ok=True)
        sorted_path = os.path.join(sorted_folder, str(i))
        sorted_imgs = glob.glob(os.path.join(sorted_path, "*.bmp"))
        for img in sorted_imgs:
            img_name = os.path.basename(img)
            shutil.copyfile(os.path.join(image_folder, img_name), os.path.join(result_path, img_name))
        



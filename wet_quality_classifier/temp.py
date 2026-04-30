import os
import glob

root_dir = fr"C:\Users\user\Desktop\Coding\Project\fingerprint\Dataset\Nasic9395_testset\test\Nasic9395_testset_test_arcface\enroll\unclassified"

if __name__ == '__main__':
    image_path = glob.glob(os.path.join(root_dir, '*.bmp'))
    for path in image_path:
        image_name = os.path.basename(path)
        names = image_name.split('_')
        new_name = names[1]
        for n in names[2:]:
            new_name = new_name + '_' + n
        os.rename(path, os.path.join(root_dir, new_name))
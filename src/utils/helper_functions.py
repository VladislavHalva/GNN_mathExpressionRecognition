# ###
# Mathematical expression recognition tool.
# Written as a part of masters thesis at VUT FIT Brno, 2022

# Author: Vladislav Halva
# Login: xhalva04
# ###

import glob
import os
import shutil


def cpy_images_inkml_gt(orig_images_root, orig_inkmls_root, tgt_images_root, tgt_inkmls_root):
    """
    Copies InkML files associated with image files in tgt_images_root to tgt_inkmls_root from orig_inkmls_root folder.
    """
    for subdir, _, files in os.walk(tgt_images_root):
        for file in files:
            image_file = file
            file_name = '.'.join(file.split('.')[:-1])
            orig_image_path = glob.glob(orig_images_root + "/**/" + image_file, recursive=True)
            if len(orig_image_path) == 1:
                orig_image_path = orig_image_path[0]
                orig_image_relative_from_root = os.path.relpath(orig_image_path, orig_images_root)
                orig_inkml_path = os.path.join(orig_inkmls_root, orig_image_relative_from_root)
                orig_inkml_path = '.'.join(orig_inkml_path.split('.')[:-1])
                orig_inkml_path += '.inkml'
                if os.path.exists(orig_inkml_path):
                    tgt_inkml_path = os.path.join(tgt_inkmls_root, file_name + '.inkml')
                    shutil.copyfile(orig_inkml_path, tgt_inkml_path)
                else:
                    print('inkml does not exists')
            else:
                print('not found or multiple with same name: ' + image_file)
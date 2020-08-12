import os
import pickle
from PIL import Image
import json
import argparse
import sys
import cv2

# fix for local importing
file_dir = os.path.dirname(__file__)
sys.path.append(file_dir)

from DFLJPG import DFLJPG
from DFLPNG import DFLPNG


def get_file_list(input_dir):
    """ Return list of images at specified location """
    result = []
    extensions = [".jpg", ".png", ".jpeg"]
    for root, _, files in os.walk(input_dir + '/'):
        for file in files:
            if os.path.splitext(file)[1].lower() in extensions:
                result.append(os.path.join(root, file))
        break
        
    return result

def load_data(file):
    if os.path.splitext(file)[1].lower() in [".jpg", ".jpeg"]:
        out = DFLJPG.load(file)
    else:
        out = DFLPNG.load(file)

    return out

def update_parent_to_self(image_folder):
    files = get_file_list(image_folder)

    for file in files:
        file_load = load_data(file)
        if file_load is not None:
            file_load.embed_and_set(file, source_filename=file, source_landmarks=file_load.get_landmarks(), source_rect=[0, 0, 256, 256])
            # print (file)
            sys.stdout.write(file + '\n')

def copy_to_file(image_folder, copy_folder):
    files = get_file_list(image_folder)
    for file in files:
        file_load = load_data(file)
        if file_load is not None:
            copy_file = os.path.join(copy_folder, os.path.basename(file))
            pre, ext = os.path.splitext(copy_file)
            copy_file = pre + '.png'
            if (os.path.isfile(copy_file) == False):
                copy_file = pre + '.jpg'
                if (os.path.isfile(copy_file) == False):
                    continue

            if os.path.splitext(copy_file)[1].lower() in [".jpg", ".jpeg"]:
                DFLJPG.embed_data(copy_file, 
                    face_type=file_load.get_face_type(),
                    source_filename=file_load.get_source_filename(),
                    source_landmarks=file_load.get_source_landmarks(),
                    landmarks=file_load.get_landmarks(),
                    source_rect=file_load.get_source_rect(),
                    image_to_face_mat=file_load.get_image_to_face_mat(),
                    eyebrows_expand_mod=file_load.get_eyebrows_expand_mod(),
                    seg_ie_polys=file_load.get_seg_ie_polys(),
                    xseg_mask=file_load.get_xseg_mask()
                    )
            else:
                DFLPNG.embed_data(copy_file, 
                    face_type=file_load.get_face_type(),
                    source_filename=file_load.get_source_filename(),
                    source_landmarks=file_load.get_source_landmarks(),
                    landmarks=file_load.get_landmarks(),
                    source_rect=file_load.get_source_rect(),
                    eyebrows_expand_mod=file_load.get_eyebrows_expand_mod(),
                    image_to_face_mat=file_load.get_image_to_face_mat()
                    )
        sys.stdout.write(file + '\n')

def convert_png_to_jpg(png_path, jpg_path):
    png_images = get_file_list(png_path)

    for file in png_images:
        file_load = load_data(file)
        if file_load is None:
            print ('No file data for file' + file)
            continue

        file_mat = cv2.imread(file)
        pre, ext = os.path.splitext(os.path.basename(file))
        copy_file = os.path.join(jpg_path, pre + '.jpg')
        cv2.imwrite(copy_file, file_mat, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        DFLJPG.embed_data(copy_file, 
            face_type=file_load.get_face_type(),
            source_filename=file_load.get_source_filename(),
            source_landmarks=file_load.get_source_landmarks(),
            landmarks=file_load.get_landmarks(),
            source_rect=file_load.get_source_rect(),
            image_to_face_mat=file_load.get_image_to_face_mat()
            )
        print ('Done: ' + copy_file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("input_folder")
    parser.add_argument("--save", action='store_true')
    parser.add_argument("--utoself", action='store_true') # updates file name to self dont use
    parser.add_argument("--copy")
    parser.add_argument("--jpg_folder")
    args = parser.parse_args()

    if args.jpg_folder is not None:
        convert_png_to_jpg(args.input_folder, args.jpg_folder)
        exit(0)

    if args.utoself == True:
        update_parent_to_self(args.input_folder)
        exit(0)

    if args.copy is not None:
        copy_to_file(args.input_folder, args.copy)
        exit(0)
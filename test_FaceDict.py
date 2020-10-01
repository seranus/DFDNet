import os
from options.test_options import TestOptions
from models import create_model
from util.visualizer import save_images
import numpy as np
import math
from PIL import Image
import torchvision.transforms as transforms
import torch
import random
import cv2
import json
import argparse
from util import util
from tqdm import tqdm
# from util import html
from skimage import transform as trans
from skimage import io
from pathlib import Path

def align_and_save(img_path, save_path, save_input_path, save_param_path, upsample_scale=2):
    out_size = (512, 512) 
    img = dlib.load_rgb_image(img_path)
    h,w,_ = img.shape
    source = get_5_points(img) 
    if source is None: #
        print('\t################ No face is detected')
        return
    tform = trans.SimilarityTransform()                                                                                                                                                  
    tform.estimate(source, reference)
    M = tform.params[0:2,:]
    crop_img = cv2.warpAffine(img, M, out_size)
    io.imsave(save_path, crop_img) #save the crop and align face
    io.imsave(save_input_path, img) #save the whole input image
    tform2 = trans.SimilarityTransform()  
    tform2.estimate(reference, source*upsample_scale)
    # inv_M = cv2.invertAffineTransform(M)
    np.savetxt(save_param_path, tform2.params[0:2,:],fmt='%.3f') #save the inverse affine parameters
    
def reverse_align(input_path, face_path, param_path, save_path, upsample_scale=2):
    out_size = (512, 512) 
    input_img = dlib.load_rgb_image(input_path)
    h,w,_ = input_img.shape
    face512 = dlib.load_rgb_image(face_path)
    inv_M = np.loadtxt(param_path)
    inv_crop_img = cv2.warpAffine(face512, inv_M, (w*upsample_scale,h*upsample_scale))
    mask = np.ones((512, 512, 3), dtype=np.float32) #* 255
    inv_mask = cv2.warpAffine(mask, inv_M, (w*upsample_scale,h*upsample_scale))
    upsample_img = cv2.resize(input_img, (w*upsample_scale, h*upsample_scale))
    inv_mask_erosion_removeborder = cv2.erode(inv_mask, np.ones((2 * upsample_scale, 2 * upsample_scale), np.uint8))# to remove the black border
    inv_crop_img_removeborder = inv_mask_erosion_removeborder * inv_crop_img
    total_face_area = np.sum(inv_mask_erosion_removeborder)//3
    w_edge = int(total_face_area ** 0.5) // 20 #compute the fusion edge based on the area of face
    erosion_radius = w_edge * 2
    inv_mask_center = cv2.erode(inv_mask_erosion_removeborder, np.ones((erosion_radius, erosion_radius), np.uint8))
    blur_size = w_edge * 2
    inv_soft_mask = cv2.GaussianBlur(inv_mask_center,(blur_size + 1, blur_size + 1),0)
    merge_img = inv_soft_mask * inv_crop_img_removeborder + (1 - inv_soft_mask) * upsample_img
    io.imsave(save_path, merge_img.astype(np.uint8))

def AddUpSample(img):
    return img.resize((512, 512), Image.BICUBIC)

# returns numpy image
def get_image_from_tensor(visuals):
    im_data = visuals['fake_A']
    im = util.tensor2im(im_data)
    return im

def landmark_68_to_5(landmarks):
    lan_5 = np.array([landmarks[45], landmarks[42], landmarks[36], landmarks[39], landmarks[34]])
    return lan_5

def get_part_location(landmarks, image):
    Landmarks = landmarks

    width, height = image.size
    if width != 512 or height != 512:
        width_scale = 512.0 / width
        height_scale = 512.0 / height

        Landmarks = Landmarks * np.array([width_scale, height_scale])

    Map_LE = list(np.hstack((range(17,22), range(36,42))))
    Map_RE = list(np.hstack((range(22,27), range(42,48))))
    Map_NO = list(range(29,36))
    Map_MO = list(range(48,68))
    try:
        #left eye
        Mean_LE = np.mean(Landmarks[Map_LE],0)
        L_LE = np.max((np.max(np.max(Landmarks[Map_LE],0) - np.min(Landmarks[Map_LE],0))/2,16))
        Location_LE = np.hstack((Mean_LE - L_LE + 1, Mean_LE + L_LE)).astype(int)
        #right eye
        Mean_RE = np.mean(Landmarks[Map_RE],0)
        L_RE = np.max((np.max(np.max(Landmarks[Map_RE],0) - np.min(Landmarks[Map_RE],0))/2,16))
        Location_RE = np.hstack((Mean_RE - L_RE + 1, Mean_RE + L_RE)).astype(int)
        #nose
        Mean_NO = np.mean(Landmarks[Map_NO],0)
        L_NO = np.max((np.max(np.max(Landmarks[Map_NO],0) - np.min(Landmarks[Map_NO],0))/2,16))
        Location_NO = np.hstack((Mean_NO - L_NO + 1, Mean_NO + L_NO)).astype(int)
        #mouth
        Mean_MO = np.mean(Landmarks[Map_MO],0)
        L_MO = np.max((np.max(np.max(Landmarks[Map_MO],0) - np.min(Landmarks[Map_MO],0))/2,16))
        Location_MO = np.hstack((Mean_MO - L_MO + 1, Mean_MO + L_MO)).astype(int)
    except:
        return 0
    return torch.from_numpy(Location_LE).unsqueeze(0), torch.from_numpy(Location_RE).unsqueeze(0), torch.from_numpy(Location_NO).unsqueeze(0), torch.from_numpy(Location_MO).unsqueeze(0)

def obtain_inputs(img, landmarks):
    Part_locations = get_part_location(landmarks, img)
    if Part_locations == 0:
        return 0

    A = img
    C = A
    A = AddUpSample(A)
    A = transforms.ToTensor()(A) 
    C = transforms.ToTensor()(C)
    A = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(A) #
    C = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(C) #
    return {'A':A.unsqueeze(0), 'C':C.unsqueeze(0), 'A_paths': A_paths,'Part_locations': Part_locations}
    
if __name__ == '__main__':  
    opt = TestOptions().parse()
    opt.nThreads = 1   # test code only supports nThreads = 1
    opt.batchSize = 1  # test code only supports batchSize = 1
    opt.serial_batches = True  # no shuffle
    opt.no_flip = True  # no flip
    opt.display_id = -1  # no visdom display
    opt.which_epoch = 'latest' #

    ####################################################
    ##Test Param
    #####################################################
    IsReal = 0
    opt.gpu_ids = [0]
    TestImgPath = opt.input_folder #% test image path
    opt.results_dir = opt.output_folder #save path

    model = create_model(opt)
    model.setup(opt)
    # test
    ImgNames = os.listdir(TestImgPath)
    ImgNames.sort()

    reference = np.load(os.path.join(Path(__file__).parent, 'packages', 'FFHQ_template.npy') / 2
    out_size = (512, 512) 

    for i, ImgName in enumerate(tqdm(ImgNames)):
        torch.cuda.empty_cache()
        data_input = input('%REQFILE%$' + ImgName)
        if len(data_input) < 2:
            continue

        # set numpy landmarks
        landmarks_string = json.loads(data_input)
        landmarks = np.array(landmarks_string)

        A_paths = os.path.join(TestImgPath, ImgName)
        Imgs = Image.open(A_paths).convert('RGB')
        img_width, img_height = Imgs.size

        # crop
        source = landmark_68_to_5(landmarks)
        tform = trans.SimilarityTransform()                                                                                                                                                  
        tform.estimate(source, reference)
        M = tform.params[0:2,:]
        array_img = np.array(Imgs)
        crop_img = cv2.warpAffine(array_img, M, out_size)
        transformed_points = tform(landmarks)

        image_pil = Image.fromarray(crop_img)

        data = obtain_inputs(image_pil, transformed_points)

        model.set_input(data)
        try:
            model.test()
            visuals = model.get_current_visuals()

            image_numpy = get_image_from_tensor(visuals)
            # crop back and resize
            image_numpy = cv2.warpAffine(image_numpy, M, Imgs.size, array_img, flags=cv2.WARP_INVERSE_MAP | cv2.INTER_CUBIC )

            image_pil = Image.fromarray(image_numpy)
            image_pil.save(os.path.join(opt.results_dir, ImgName))
        except Exception as e:
            print(r'%ERROR%$Error in enhancing this image: {}'.format(str(e)))

            continue
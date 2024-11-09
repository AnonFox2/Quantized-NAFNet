import cv2
import random
import numpy as np

def adjust_image_size(image, size_multiplier):
    image = image.copy()
    
    if len(image.shape) in (2, 3):
        height = image.shape[0]
        width = image.shape[1]
        
        extra_height = height % size_multiplier
        extra_width = width % size_multiplier
        
        image = image[:height - extra_height, :width - extra_width, ...]
    else:
        raise ValueError(f'Image has wrong number of dimensions: {len(image.shape)}')
    
    return image

def crop_matching_patches(high_res_imgs, low_res_imgs, patch_size, image_path, scale=1):
    if not isinstance(high_res_imgs, list):
        high_res_imgs = [high_res_imgs]
    if not isinstance(low_res_imgs, list):
        low_res_imgs = [low_res_imgs]
    
    low_res_height = low_res_imgs[0].shape[0]
    low_res_width = low_res_imgs[0].shape[1]
    high_res_height = high_res_imgs[0].shape[0]  
    high_res_width = high_res_imgs[0].shape[1]
    
    small_patch_size = patch_size // scale
    
    if high_res_height != low_res_height * scale or high_res_width != low_res_width * scale:
        raise ValueError(f'Images dont match in size. High res ({high_res_height}, {high_res_width}) is not {scale}x larger than low res ({low_res_height}, {low_res_width})')
    
    if low_res_height < small_patch_size or low_res_width < small_patch_size:
        raise ValueError(f'Low res image ({low_res_height}, {low_res_width}) is too small for patch size ({small_patch_size}, {small_patch_size}). Please remove {image_path}')
    
    start_y = random.randint(0, low_res_height - small_patch_size)
    start_x = random.randint(0, low_res_width - small_patch_size)
    
    cropped_low_res = [
        img[start_y:start_y + small_patch_size, start_x:start_x + small_patch_size, ...]
        for img in low_res_imgs
    ]
    
    high_res_y = int(start_y * scale)
    high_res_x = int(start_x * scale)
    cropped_high_res = [
        img[high_res_y:high_res_y + patch_size, high_res_x:high_res_x + patch_size, ...]
        for img in high_res_imgs
    ]
    
    if len(high_res_imgs) == 1:
        cropped_high_res = cropped_high_res[0]
    if len(low_res_imgs) == 1:
        cropped_low_res = cropped_low_res[0]
        
    return cropped_high_res, cropped_low_res

def crop_matching_patches_rectangular(high_res_imgs, low_res_imgs, patch_height, patch_width, scale, image_path):
    if not isinstance(high_res_imgs, list):
        high_res_imgs = [high_res_imgs]
    if not isinstance(low_res_imgs, list):
        low_res_imgs = [low_res_imgs]

    low_res_height = low_res_imgs[0].shape[0]  
    low_res_width = low_res_imgs[0].shape[1]
    high_res_height = high_res_imgs[0].shape[0]
    high_res_width = high_res_imgs[0].shape[1]

    small_patch_height = patch_height // scale
    small_patch_width = patch_width // scale

    start_y = random.randint(0, low_res_height - small_patch_height)
    start_x = random.randint(0, low_res_width - small_patch_width)

    cropped_low_res = [
        img[start_y:start_y + small_patch_height, start_x:start_x + small_patch_width, ...]
        for img in low_res_imgs
    ]

    high_res_y = int(start_y * scale)
    high_res_x = int(start_x * scale)
    cropped_high_res = [
        img[high_res_y:high_res_y + patch_height, high_res_x:high_res_x + patch_width, ...]
        for img in high_res_imgs
    ]

    if len(high_res_imgs) == 1:
        cropped_high_res = cropped_high_res[0]
    if len(low_res_imgs) == 1:
        cropped_low_res = cropped_low_res[0]
        
    return cropped_high_res, cropped_low_res

def apply_transforms(images, do_flip_horizontal=True, do_rotate=True, flows=None, return_status=False, do_flip_vertical=False):
    should_flip_horizontal = do_flip_horizontal and random.random() < 0.5
    if do_flip_vertical or do_rotate:
        should_flip_vertical = random.random() < 0.5
    should_rotate = do_rotate and random.random() < 0.5

    def transform_image(img):
        if should_flip_horizontal:
            cv2.flip(img, 1, img)
            if img.shape[2] == 6:
                img = img[:,:,[3,4,5,0,1,2]].copy()
        if should_flip_vertical:
            cv2.flip(img, 0, img)
        if should_rotate:
            img = img.transpose(1, 0, 2)
        return img

    def transform_flow(flow):
        if should_flip_horizontal:
            cv2.flip(flow, 1, flow)
            flow[:, :, 0] *= -1
        if should_flip_vertical:
            cv2.flip(flow, 0, flow)
            flow[:, :, 1] *= -1
        if should_rotate:
            flow = flow.transpose(1, 0, 2)
            flow = flow[:, :, [1, 0]]
        return flow

    if not isinstance(images, list):
        images = [images]
    transformed_images = [transform_image(img) for img in images]
    if len(transformed_images) == 1:
        transformed_images = transformed_images[0]

    if flows is not None:
        if not isinstance(flows, list):
            flows = [flows]
        transformed_flows = [transform_flow(flow) for flow in flows]
        if len(transformed_flows) == 1:
            transformed_flows = transformed_flows[0]
        return transformed_images, transformed_flows
    else:
        if return_status:
            return transformed_images, (should_flip_horizontal, should_flip_vertical, should_rotate)
        else:
            return transformed_images

def rotate_image(image, angle, center=None, scale=1.0):
    height, width = image.shape[:2]

    if center is None:
        center = (width // 2, height // 2)

    rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
    rotated = cv2.warpAffine(image, rotation_matrix, (width, height))
    return rotated
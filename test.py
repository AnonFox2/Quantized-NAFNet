import torch
import argparse
import cv2
import numpy as np
import matplotlib.pyplot as plt
from deblurring_model import DeblurringModel
from utils import imfrombytes, img2tensor, padding, tensor2img, imwrite

def parse_args():
    parser = argparse.ArgumentParser(description="Image Deblurring Script")
    parser.add_argument('--input_img', type=str, required=True, help="Path to the input image")
    parser.add_argument('--output_img', type=str, required=True, help="Path to save the output image")
    parser.add_argument('--weights', type=str, default=None, help='Path to the pretrained weights (.pth file)')
    return parser.parse_args()

def resize_image(image, max_size=(800, 800)):
    height, width = image.shape[:2]
    max_height, max_width = max_size
    scale = min(max_width / width, max_height / height)
    if scale < 1:
        image = cv2.resize(image, (int(width * scale), int(height * scale)), interpolation=cv2.INTER_AREA)
    return image

def main():
    args = parse_args()
    input_img_path = args.input_img
    output_img_path = args.output_img

    # Load and preprocess the image
    try:
        img = cv2.imread(input_img_path, cv2.IMREAD_UNCHANGED)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = img_rgb.astype(np.float32) / 255.0
    except:
        raise Exception(f"Path {input_img_path} not working")

    img_tensor = img2tensor(img, bgr2rgb=False, float32=False)

    # Load model and perform inference
    opt = {'img_path': {'input_img': input_img_path, 'output_img': output_img_path}}
    model = DeblurringModel(training="val", load_path=args.weights)
    model.feed_data(data={'lq': img_tensor.unsqueeze(dim=0)})
    model.test()

    # Process and save the output image
    visuals = model.get_current_visuals()
    dblr_img = tensor2img([visuals['result']])
    imwrite(dblr_img, output_img_path)

    print(f"Inference {input_img_path} .. finished. Saved to {output_img_path}")

    # Resize images for display
    img_resized = resize_image(img, max_size=(800, 800))
    dblr_img_resized = resize_image(dblr_img, max_size=(800, 800))
    dblr_img_resized = cv2.cvtColor(dblr_img_resized, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(18, 9))
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0)

    plt.subplot(1, 2, 1)
    plt.imshow(img_resized)
    plt.title("Input Image")
    plt.axis("off")

    plt.subplot(1, 2, 2)
    plt.imshow(dblr_img_resized)
    plt.title("Output Image")
    plt.axis("off")

    plt.show(block=False)
    plt.pause(4)  # Display for 4 seconds
    plt.close()

if __name__ == '__main__':
    main()

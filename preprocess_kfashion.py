import PIL.Image as Image
import numpy as np
import cv2
import argparse
import os

from process import load_seg_model
from preprocess import preprocess_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Help to set arguments for Cloth Segmentation.')
    parser.add_argument('--input_path', type=str, help='Path to the input image')
    parser.add_argument('--cuda', action='store_true', help='Enable CUDA (default: False)')
    parser.add_argument('--checkpoint_path', type=str, default='model/cloth_segm.pth', help='Path to the checkpoint file')
    parser.add_argument('--save_path', type=str, default='preprocessed_image/', help='Path to save preprocessed image')
    args = parser.parse_args()
    
    device = 'cuda:0' if args.cuda else 'cpu'
    
    # load model
    model = load_seg_model(args.checkpoint_path, device=device)

    styles = []
    
    # get style names
    if os.path.isdir(args.input_path):
        styles = os.listdir(args.input_path)
        print(styles)
    # open one image
    else:
        styles.append(args.input_path)
        
    # create output directory
    if not os.path.isdir(args.save_path):
        os.mkdir(args.save_path)
    
    # preprocess image one by one
    for s in styles:
        print("Processing ", s)
        
        #create output file for each style
        # save_path = os.path.join(args.save_path, s)
        save_path = os.path.join(args.save_path, s)
        print("save_path: ", save_path)
        if not os.path.isdir(save_path):
            os.mkdir(save_path)
        
        input_path = os.path.join(args.input_path, s)
        # input_path = s
        print("input_path: ", input_path)
        target = os.listdir(input_path)
            
        for t in target:
            image = Image.open(os.path.join(args.input_path, s, t)).convert('RGB')
            image_id = os.path.splitext(os.path.basename(t))[0]
            print(image_id)
            preprocess_image(args, s, model, image, image_id, device)
    
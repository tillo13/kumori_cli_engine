from configs import initial_image, OUTPUT_FOLDER_NAME, choose_random_model


import os
import cv2
import math
#import spaces
import torch
import random
import numpy as np
import argparse
from datetime import datetime

import PIL
from PIL import Image, ImageOps
import shutil
from pathlib import Path
import diffusers
from diffusers.utils import load_image
from diffusers.models import ControlNetModel 
import insightface
from insightface.app import FaceAnalysis
from style_template import styles
from pipeline_stable_diffusion_xl_instantid import StableDiffusionXLInstantIDPipeline

import argparse
import re
import sys
import logging
import numpy as np

# Assumed existing logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# global variable set first before randomizing model choice
HUGGINGFACE_MODEL = 'RunDiffusion/Juggernaut-X-v10'

MAX_SEED = np.iinfo(np.int32).max
#device = "cuda" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16
if torch.backends.mps.is_available():
  device = "mps"
  torch_dtype = torch.float32
elif torch.cuda.is_available():
  device = "cuda"
else:
  device = "cpu"
STYLE_NAMES = list(styles.keys())
DEFAULT_STYLE_NAME = "Watercolor"

# download checkpoints
from huggingface_hub import hf_hub_download
from pathlib import Path

# Check if the configuration file exists locally
config_file = Path("./checkpoints/ControlNetModel/config.json")
if config_file.exists():
    print("FYI:::::Found local configuration file, just using it.")
else:
    # If it doesn't exist, download it
    print("FYI:::::Downloading configuration file as no local one present...")
    try:
        hf_hub_download(repo_id="InstantX/InstantID", 
                        filename="ControlNetModel/config.json", 
                        local_dir="./checkpoints")
    except Exception as e:
        print(f"An error occurred while attempting to download the configuration file: {e}")

# Check if the diffusion_pytorch_model.safetensors file exists locally
safetensors_file = Path("./checkpoints/ControlNetModel/diffusion_pytorch_model.safetensors")
if safetensors_file.exists():
    print("FYI:::::Found local diffusion_pytorch_model.safetensors file, just using it.")
else:
    # If it doesn't exist, download it
    print("FYI:::::Downloading diffusion_pytorch_model.safetensors file as no local one present...")
    try:
        hf_hub_download(repo_id="InstantX/InstantID", 
                        filename="ControlNetModel/diffusion_pytorch_model.safetensors", 
                        local_dir="./checkpoints")
    except Exception as e:
        print(f"An error occurred while attempting to download the diffusion_pytorch_model.safetensors file: {e}")

# Check if the ip-adapter.bin file exists locally
adapter_file = Path("./checkpoints/ip-adapter.bin")
if adapter_file.exists():
    print("FYI:::::Found local ip-adapter.bin file, just using it.")
else:
    # If it doesn't exist, download it
    print("FYI:::::Downloading ip-adapter.bin file as no local one present...")
    try:
        hf_hub_download(repo_id="InstantX/InstantID", 
                        filename="ip-adapter.bin", 
                        local_dir="./checkpoints")
    except Exception as e:
        print(f"An error occurred while attempting to download the ip-adapter.bin file: {e}")

# Load face encoder
app = FaceAnalysis(name='antelopev2', root='./', providers=['CPUExecutionProvider'])
app.prepare(ctx_id=0, det_size=(640, 640))

# Path to InstantID models
face_adapter = f'./checkpoints/ip-adapter.bin'
controlnet_path = f'./checkpoints/ControlNetModel'

# Load pipeline
controlnet = ControlNetModel.from_pretrained(controlnet_path, torch_dtype=torch_dtype)

def randomize_seed_fn(seed: int, randomize_seed: bool) -> int:
    if randomize_seed:
        seed = random.randint(0, MAX_SEED)
    return seed

def convert_from_cv2_to_image(img: np.ndarray) -> Image:
    return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

def convert_from_image_to_cv2(img: Image) -> np.ndarray:
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def draw_kps(image_pil, kps, color_list=[(255,0,0), (0,255,0), (0,0,255), (255,255,0), (255,0,255)]):
    stickwidth = 4
    limbSeq = np.array([[0, 2], [1, 2], [3, 2], [4, 2]])
    kps = np.array(kps)

    w, h = image_pil.size
    out_img = np.zeros([h, w, 3])

    for i in range(len(limbSeq)):
        index = limbSeq[i]
        color = color_list[index[0]]
        x = kps[index][:, 0]
        y = kps[index][:, 1]
        length = ((x[0] - x[1]) ** 2 + (y[0] - y[1]) ** 2) ** 0.5
        angle = math.degrees(math.atan2(y[0] - y[1], x[0] - x[1]))
        polygon = cv2.ellipse2Poly((int(np.mean(x)), int(np.mean(y))), (int(length / 2), stickwidth), int(angle), 0, 360, 1)
        out_img = cv2.fillConvexPoly(out_img.copy(), polygon, color)
    out_img = (out_img * 0.6).astype(np.uint8)

    for idx_kp, kp in enumerate(kps):
        color = color_list[idx_kp]
        x, y = kp
        out_img = cv2.circle(out_img.copy(), (int(x), int(y)), 10, color, -1)

    out_img_pil = Image.fromarray(out_img.astype(np.uint8))
    return out_img_pil

def apply_style(style_name: str, positive: str, negative: str = "") -> tuple[str, str]:
    p, n = styles.get(style_name, styles[DEFAULT_STYLE_NAME])
    return p.replace("{prompt}", positive), n + ' ' + negative

def pad_25_percent(image_path):
    # Open the original image
    image = Image.open(image_path)

    # Calculate the padding size
    padding_width = image.width * 0.25
    padding_height = image.height * 0.25

    # Create a new image with white background that is larger to include the padding
    new_width = image.width + int(padding_width * 2)  # Adding padding to both sides
    new_height = image.height + int(padding_height * 2)  # Adding padding to top and bottom
    new_image = Image.new("RGB", (new_width, new_height), "white")

    # Paste the original image onto the center of the new image thus creating the padding effect
    new_image.paste(image, (int(padding_width), int(padding_height)))

    # Save the padded image
    image_dir, image_file = os.path.split(image_path)
    
    new_image_file = "padded_" + image_file
    new_image_path = os.path.join(image_dir, new_image_file)
    new_image.save(new_image_path)

    print(f"Padded image saved as {new_image_path}")

    # Move the original image to a 'needs_padding' folder
    needs_padding_folder = Path(image_dir, "needs_padding")
    # Create the needs_padding folder if it doesn't exist
    needs_padding_folder.mkdir(exist_ok=True)
    
    # Construct the new path for the original image in the 'needs_padding' folder
    moved_image_path = needs_padding_folder / image_file
    
    # Move the original image to the 'needs_padding' folder
    shutil.move(image_path, moved_image_path)
    
    print(f"Original image {image_path} moved to {moved_image_path}.")

def generate_image(face_image, pose_image, prompt, negative_prompt, style_name, enhance_face_region, num_steps, identitynet_strength_ratio, adapter_strength_ratio, guidance_scale, seed):
    generated_file_paths = []
    face_emb = None  # or a default value
    model_used = "ErrorModel"  # Placeholder for error cases

    # Log the state of pose_image regardless
    print(f"\nKUMORI_CLI.PY pose_image passed in: {pose_image}")
    print("=======")    

    try:
        if face_image is None:
            logging.error("KUMORI_CLI.PY: face_image is None. Cannot find any input face image! Please upload the face image")

        if prompt is None:
            prompt = "a person"

        if style_name is not None:
            print("\n=======")
            print(f"KUMORI_CLI.PY: Style passed from PREPROCESS.PY: {style_name}")
            print("\n=======")
            
            # Apply the style to the prompt
            prompt, negative_prompt = apply_style(style_name, prompt, negative_prompt)
        else:
            # Log when no specific style is being used
            print("\n=======")
            print("KUMORI_CLI.PY: No style passed in from PREPROCESS.PY, which is fine, so using no additional style...")
            print("\n=======")

        logging.info(f"Outside --Loading face image from path: {face_image[0]}")

        # Log the resizing step for the ingested image.
        logging.info(f"Resizing ingested image: {face_image[0]}")
        face_org = face_image[0]
        face_image = load_image(face_image[0])
        face_image = resize_img(face_image, output_directory='debug_image_test', image_type='ingested', max_side=max_side, min_side=min_side)
        
        face_image_cv2 = convert_from_image_to_cv2(face_image)
        height, width, _ = face_image_cv2.shape

        # Extract face features
        face_info = app.get(face_image_cv2)

        if len(face_info) == 0:
            logging.info(f"KUMORI_CLI.PY: face_info length ==0. Cannot find any face in the image! Please upload another person image")
            logging.info(f"KUMORI_CLI.PY: No face detected, attempting to pad the image: {face_org}")
            try:
                # Attempting to pad the image
                pad_25_percent(face_org)
                logging.info("Padding successful")
            except Exception as e:
                logging.error(f"Padding failed with error: {e}")

        face_info = face_info[-1]
        face_emb = face_info['embedding']
        face_kps = draw_kps(convert_from_cv2_to_image(face_image_cv2), face_info['kps'])

        if pose_image is not None:
            # Log the resizing step for the pose image.
            logging.info(f"Resizing pose image: {pose_image[0]}")
            pose_image = load_image(pose_image[0])
            pose_image = resize_img(pose_image, output_directory='debug_image_test', image_type='pose', max_side=max_side, min_side=min_side)

            pose_image_cv2 = convert_from_image_to_cv2(pose_image)
            
            face_info = app.get(pose_image_cv2)

            if len(face_info) == 0:
                logging.error("KUMORI_CLI.PY: len(face_info ==0) Cannot find any face in the reference image! Please upload another person image")

                
            face_info = face_info[-1]
            face_kps = draw_kps(pose_image, face_info['kps'])
            
            width, height = face_kps.size
        
        if enhance_face_region:
            control_mask = np.zeros([height, width, 3])
            x1, y1, x2, y2 = face_info['bbox']
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            control_mask[y1:y2, x1:x2] = 255
            control_mask = Image.fromarray(control_mask.astype(np.uint8))
        else:
            control_mask = None
        
        generator = torch.Generator(device=device).manual_seed(seed)        
        pipe.set_ip_adapter_scale(adapter_strength_ratio)

        # choose a random model from preprocess.py before submitting to the pipe
        HUGGINGFACE_MODEL = choose_random_model()

        # Print the model being used right before the pipeline is called
        print("\n=======")
        print(f"KUMORI_CLI.PY: HuggingFace model we will use right before sending to Pipeline: {HUGGINGFACE_MODEL}")
        print(f"KUMORI_CLI.PY: prompt right before sending to Pipeline: {prompt}")
        print("=======")


        # Invoke the pipeline with the specified parameters
        images = pipe(
            prompt=prompt,
            negative_prompt=negative_prompt,
            image_embeds=face_emb,
            image=face_kps,
            control_mask=control_mask,
            controlnet_conditioning_scale=float(identitynet_strength_ratio),
            num_inference_steps=num_steps,
            guidance_scale=guidance_scale,
            height=height,
            width=width,
            generator=generator
        ).images[0]

        print("\n=======")            
        print("KUMORI_CLI.PY: Yay! Image generation successful on first try/catch!")
        print("\n=======")

        # Check if the output directory exists, create it if it doesn't
        output_dir_path = os.path.join(os.getcwd(), OUTPUT_FOLDER_NAME)
        os.makedirs(output_dir_path, exist_ok=True)

        # Save the generated images with a timestamp
        generated_file_paths = []  # Initialize an empty list to store generated image file paths

        # Normalizing HUGGINGFACE_MODEL for filename
        normalized_hf_model_name = re.sub(r'\W+', '_', HUGGINGFACE_MODEL)

        # Generate filename using the normalized_hf_model_name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = f"{normalized_hf_model_name}_{timestamp}.png"
        filepath = os.path.join(output_dir_path, filename)
        images.save(filepath)
        print("\n=======")
        print(f"KUMORI_CLI.PY: Output from StableDiff pipeline saved as {filepath}")
        print("\n=======")

        generated_file_paths.append(filepath)  # Store the file path

    except Exception as e:
        # Handle any exceptions that have occurred
        print(f"Error during image generation: {e}")

        # Here's a simple fallback example where we retry with lower resolution
        try:
            height, width = 512, 512  # Set smaller resolution for lower memory usage
            images = pipe(
                prompt=prompt,
                negative_prompt=negative_prompt,
                image_embeds=face_emb,
                image=face_kps,
                control_mask=control_mask,
                controlnet_conditioning_scale=float(identitynet_strength_ratio),
                num_inference_steps=num_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator
            ).images[0]
        except Exception as fallback_exception:
            # If the fallback also fails, handle the exception such as logging an error or notifying the user
            print(f"Error during fallback image generation: {fallback_exception}")
            return None, None, []

    print(f"HuggingFace model returning at the end of generate_image within KUMORI_CLI.PY: {HUGGINGFACE_MODEL}")
    return images, None, generated_file_paths, HUGGINGFACE_MODEL 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--inbrowser', action='store_true', help='Open in browser')
    parser.add_argument('--server_port', type=int, default=7860, help='Server port')
    parser.add_argument('--share', action='store_true', help='Share the Gradio UI')
    parser.add_argument('--medvram', action='store_true', help='Medium VRAM settings')
    parser.add_argument('--lowvram', action='store_true', help='Low VRAM settings')

    # Use HUGGINGFACE_MODEL as the default value for the --model_path argument
    parser.add_argument('--model_path', type=str, default=HUGGINGFACE_MODEL, help='Base model path')

    args = parser.parse_args()

    # Now, args.model_path will use the default HUGGINGFACE_MODEL if no model path is provided by the user
    base_model_path = args.model_path

    # Default values for max_side and min_side
    max_side = 1280
    min_side = 1024

    # Adjust max_side and min_side based on the arguments
    if args.medvram:
        max_side, min_side = 1024, 832
    elif args.lowvram:
        max_side, min_side = 832, 640

    # Display the current resolution settings
    print(f"Current resolution settings for both POSE and INCOMING image: max_side = {max_side}, min_side = {min_side}")

    def save_debug_image(image, directory, base_filename, suffix, image_type):
        os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
        debug_filename = f"{image_type}_{base_filename}_{suffix}.png"
        debug_full_path = os.path.join(directory, debug_filename)
        image.save(debug_full_path)
        #logging.info(f"Saved {image_type} debug image: {debug_full_path}")

    def resize_img(input_image, output_directory='debug_image_test', image_type='ingested', max_side=1280, min_side=1024, pad_to_max_side=False, mode=Image.LANCZOS, base_pixel_number=64):
        # Generate a base filename for debug images
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_filename = f"image_{timestamp}"
        
        # Save the original image as it comes in, mark it as either 'ingested' or 'pose'
        logging.info(f"Original {image_type} image size: {input_image.size}")

        w, h = input_image.size
        aspect_ratio = w / h
        logging.info(f"Original aspect ratio: {aspect_ratio:.2f}")

        # Apply original resizing logic first
        ratio = min(min_side / min(h, w), max_side / max(h, w))
        new_size = (int(w * ratio), int(h * ratio))
        input_image = input_image.resize(new_size, Image.BILINEAR)

        # Snap to the nearest base_pixel requirements
        w_resize_new = (new_size[0] // base_pixel_number) * base_pixel_number
        h_resize_new = (new_size[1] // base_pixel_number) * base_pixel_number
        input_image = input_image.resize((w_resize_new, h_resize_new), mode)
        logging.info(f"Image resized to: {input_image.size}")

        # Apply padding if requested
        if pad_to_max_side:
            res = np.ones([max_side, max_side, 3], dtype=np.uint8) * 255
            offset_x = (max_side - w_resize_new) // 2
            offset_y = (max_side - h_resize_new) // 2
            res[offset_y:offset_y + h_resize_new, offset_x:offset_x + w_resize_new] = np.array(input_image)
            input_image = Image.fromarray(res)
            logging.info(f"Padding applied. New image size: {input_image.size}")

        return input_image  # Return the resized image

    # Display only the arguments currently in use
    print("Arguments currently in use:")
    default_values = parser.parse_args([])
    for arg, value in vars(args).items():
        if value != getattr(default_values, arg):
            print(f"  {arg}: {value}")
            
    pipe = StableDiffusionXLInstantIDPipeline.from_pretrained(
    base_model_path,
    controlnet=controlnet,
    torch_dtype=torch_dtype,
    safety_checker=None,
    feature_extractor=None,
)

try:
    if device == 'mps':
        logging.info("Configuring pipeline for MPS device...")
        pipe.to("mps", torch_dtype)
        pipe.enable_attention_slicing()
        logging.info("Pipeline successfully configured for MPS.")
    elif device == 'cuda':
        logging.info("Moving pipeline to CUDA...")
        pipe.cuda()  # Move pipeline to CUDA
        logging.info("Pipeline successfully moved to CUDA.")
except Exception as e:
    logging.error(f"Error during pipeline configuration for device {device}: {e}")
    sys.exit("Stopping execution due to previous error")

try:
    logging.info("Loading IP adapter...")
    pipe.load_ip_adapter_instantid(face_adapter)
    logging.info("IP adapter loaded successfully.")
except Exception as e:
    logging.error(f"Error loading IP adapter: {e}")
    sys.exit("Stopping execution due to previous error")

try:
    if device == 'mps' or device == 'cuda':
        logging.info("Configuring image_proj_model and unet for the specified device...")
        pipe.image_proj_model.to(device)
        pipe.unet.to(device)
        logging.info("Components successfully configured for the specified device.")
except Exception as e:
    logging.error(f"Error during component configuration for device {device}: {e}")
    sys.exit("Stopping execution due to previous error")

try:
    try:
        logging.info("Starting image preprocessing and generation process...")
        initial_image(generate_image)  # Here we pass the generate_image function to pre-process images.
    except Exception as e:
        logging.error(f"Error during image processing: {e}")
        # For now, let's simply log a message and continue ==but also exapnding the canvas using PILLOW would work here, but TBD...
        logging.info("Attempting to continue with the next image or process...")
    logging.info("All images and loops have been completed!")
except Exception as e:
    logging.error(f"Error during image generation or demo launch: {e}")
    sys.exit("Stopping execution due to previous error")
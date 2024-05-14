import os
import random
import csv
import gc
import glob
from datetime import datetime
import time
from pathlib import Path
from style_template import style_list
from PIL import Image, ImageOps
import shutil
import re

#created fileset unique to preprocess
from gender_detect import detect_gender_and_age

#detect to which pose the image of the person best aligns...
from estimate_similar_faces import main as compare_faces_main

#for the padding function if needed
from shutil import move
from os import path, makedirs

#########FEMALE_PROMPT VS. MALE_PROMPt WILL BE DETERMINED VIA GENDER_DETECT.PY, IF FEMALE_PROMPT BLANK, WILL DEFAULT TO MALE_PROMPT###################
MALE_PROMPT="""
male noble, high fantasy art, imppressionist painting, brom, john howe, alan lee, hobbit art, luis royo, tern expression,  wearing golden stag antler crown, fur-lined cloak shades of white, greys, light brown, wolf pelt, robe over right shoulder, chest adorned with mystical medallions, holding wooden staff, rough finish, glowing orb top, curled grip
"""

FEMALE_PROMPT = """
artistic digital painting, a woman with beautiful blue hair, orange beanie and scarf, eyes with full lashes. Warm and cool tones, starry background. Stars art style, water drops, romantic, intense lines, gongbi, loose paint, stylize 750, AR 71:98
"""

NEGATIVE_PROMPT="""
Realism, poorly drawn hands/face, bad proportions. Poor quality: bad eyes, text, cropped, out of frame, jpeg artifacts.
"""
#########LOOPS#########################
#how many times should we process this image set?
NUMBER_OF_LOOPS = 250 
#########RANDOM STYLE FROM STYLE_TEMPLATES.PY#########################
# If this is True, the PERC_OF_STYLE is used, otherwise it will be ignored
USE_RANDOM_STYLE = True
# Define the global variable for the percentage chance of choosing a style so if 40 here, then there will be a 40% chance it will pick a random style_list.py style.
PERC_OF_STYLE = 20

#########HUGGINGFACE MODEL LIST########
#will default to FIRST entry in GLOBAL_MODEL_LIST if False otherwise will just randomly pick a model on EACH image (which is good for variation tests).
RANDOM_MODEL_ENABLED = True 

# Validated working models that work with InstantID, set to False to not use
GLOBAL_MODEL_LIST = {
    'wangqixun/YamerMIX_v8': True,
    'RunDiffusion/juggernaut-xl-v8': False,
    'RunDiffusion/Juggernaut-X-v10': True,
    'RunDiffusion/Juggernaut-XL-Lightning': True,
    'stablediffusionapi/realism-engine-sdxl-v30': False,
}

###############POSE CONFIGURATION#################
# Seting PRESET_POSE_ENABLED to False will give the StableDiffusion model more randomness in the pose choices, but also means the image might not align as cleanly.  From what I've noticed it does hit right maybe 70% of the time, but strongly recommend setting a pose or 2 that you like, instead of tossing it to the robots to decide...
PRESET_POSE_ENABLED = True

# Validated working poses in the folder, set to False to not use in randomization
GLOBAL_POSE_LIST = {
    "poses/anyone_pose1.webp": True,
}

GLOBAL_MALE_POSE_LIST = {
    "poses/male_jedi1.webp": False,
    "poses/male_king.jpeg": True,
}

GLOBAL_FEMALE_POSE_LIST = {
    "poses/female_painting_bluehair.webp": True,
    "poses/female_superhero2.webp": False,
}

#This sets the overall perc that you're comfortable assuming a gender_detect.py result will assign the value.  I've seen so far that it's pretty good at detecting, so this is more a failsafe in case it worth lowering to be sure.
GLOBAL_FEMALE_POSE_PERC=60

##############LOGGING CONFIGURATION###################
# Setting this will log details of outputs to a generation_log.csv file in the same directory
LOGGING_ENABLED = True
LOG_FILENAME = 'generation_log.csv'
logfile_path = os.path.join(os.getcwd(), LOG_FILENAME)

INPUT_FOLDER_NAME = 'incoming_images'
OUTPUT_FOLDER_NAME = 'generated_images'

##############INSTANTID SPECIFIC CONFIGURATIONS############
# IDENTITYNET_STRENGTH_RATIO_RANGE: This range is critical for capturing and preserving the unique semantic facial information, such as eye color or nose shape. Based on InstantID's emphasis on zero-shot, high-fidelity identity preservation with single images, we choose this higher range to ensure the generated image retains the distinct characteristics of the individual's identity as closely as possible. This aligns with InstantID's capability of achieving high fidelity without the need for extensive fine-tuning or multiple reference images like LoRA.
IDENTITYNET_STRENGTH_RATIO_RANGE = (1.5, 2.2)

# ADAPTER_STRENGTH_RATIO_RANGE: This range influences the extent to which the model captures and replicates the intricate details from the reference facial image, thanks to the IP-Adapter's role in encoding these details and offering spatial control. A balanced approach aims to enhance detail fidelity while avoiding excessive saturation, drawing on InstantID's strength in seamlessly blending styles and capturing personal identity features.
ADAPTER_STRENGTH_RATIO_RANGE = (0.8, 1.2)

# NUM_INFERENCE_STEPS_RANGE: A higher number of steps allows for a more detailed, refined image generation process, aligning with InstantID's approach to generating personalized, high-quality images efficiently. Given the emphasis on accuracy and detail over speed, this range ensures the model has ample opportunity to process and incorporate the nuances of the individual's identity, as well as intricate style details.
NUM_INFERENCE_STEPS_RANGE = (25, 60)  

# GUIDANCE_SCALE_RANGE: This parameter is fine-tuned to ensure a strong alignment of the generated image with textual prompts while preserving the unique identity attributes captured by IdentityNet and ControlNet. The chosen range reflects InstantID's capability for detailed, faithful replication of identity attributes within various stylistic interpretations, ensuring that the final image is not only stylistically coherent but also an accurate reflection of the individual’s identity.
GUIDANCE_SCALE_RANGE = (2.5, 9.0)    

########RANDOMIZERS##############
def choose_random_pose(pose_list):
    # Filter active poses based on the provided list
    active_poses = [pose for pose, is_active in pose_list.items() if is_active]
    
    if PRESET_POSE_ENABLED and active_poses:
        return random.choice(active_poses)
    else:
        # return the first active pose from the list as a default if RANDOM_POSE_ENABLED is False
        if active_poses:
            return active_poses[0]
        else:
            print("No active poses available in the directory, all good, Override will catch it.  Proceeding without a pose.")
            return None

# Dynamically create the STYLES list from imported style_list
STYLES = [style["name"] for style in style_list]
def choose_random_style():
    # Check if USE_RANDOM_STYLE is False, then bypass the random choice
    if not USE_RANDOM_STYLE:
        print("USE_RANDOM_STYLE is False: Skipping random style selection")
        return None
    
    # Generate a random number between 1 and 100
    chance = random.randint(1, 100)

    # Check if the random number falls within the percentage chance of choosing a style
    if chance <= PERC_OF_STYLE:
        chosen_style = random.choice(STYLES)
        print("\n*******")
        print(f"PERC_OF_STYLE ({PERC_OF_STYLE}%) chance hit randomly choosing the number {chance}: Choosing a random style -> {chosen_style}")
        return chosen_style
    else:
        print("\n*******")
        print(f"PERC_OF_STYLE ({PERC_OF_STYLE}%) chance missed choosing the number {chance}: Using 'no style'")
        return None

def choose_random_model():
    # Filter active models based on the GLOBAL_MODEL_LIST
    active_models = [model for model, is_active in GLOBAL_MODEL_LIST.items() if is_active]
    
    if RANDOM_MODEL_ENABLED:
        if active_models:  # Check if there are any active models
            return random.choice(active_models)
        else:
            raise ValueError("No active models available.")  # Or handle this as you see fit
    else:
        #return the first active model from the list as a default if RANDOM_MODEL_ENABLED is False
        if active_models:
            return active_models[0]
        else:
            raise ValueError("No active models available.")  # Or return a specific hardcoded default model

def get_modified_model_name(model_name):
    ns_pattern = re.compile(r'-?(invalid_characters_here)')
    # Use regex to substitute matched patterns with an empty string, effectively removing them
    clean_model_name = re.sub(ns_pattern, '', model_name)    
    # Clean any possible trailing or leading hyphens or underscores after removal
    clean_model_name = clean_model_name.strip('-_')    

    return clean_model_name
########RANDOMIZERS END OF DAYS, OR SECTION##############

########LOG FILE LOGIC##############
def move_old_logs_to_log_folder():
    logs_folder_path = os.path.join(os.getcwd(), 'LOGS')
    os.makedirs(logs_folder_path, exist_ok=True)  # Create LOGS directory if doesn't exist

    for log_file in glob.glob('generation_log*.csv'):
        try:
            # Build the path for the destination
            destination_path = os.path.join(logs_folder_path, log_file)

            # Check if a file with the same name already exists
            if os.path.isfile(destination_path):
                # Get the current datetime in the specified format and attach to filename
                current_datetime = datetime.now().strftime('%Y%m%d_%H%M%S')
                file_name, file_ext = os.path.splitext(log_file)
                new_file_name = f"{file_name}_{current_datetime}{file_ext}"
                destination_path = os.path.join(logs_folder_path, new_file_name)

            # Move the file
            shutil.move(log_file, destination_path)
            print(f"Moved '{log_file}' to '{os.path.basename(destination_path)}'.")
        except Exception as e:
            print(f"Error moving '{log_file}': {e}")
            
def log_image_generation_details(
    basename, subdir_name, new_file_name, identitynet_strength_ratio, adapter_strength_ratio,
    num_inference_steps, guidance_scale, seed, success, error_message, style_name,
    prompt, negative_prompt, loop_time_taken, current_timestamp, huggingface_model, 
    detection_results, logfile_path,chosen_pose, pose_filename='', eye_distance=0, nose_mouth_distance=0, distance_difference=0):
    
    # Ensure the directory exists for the log file
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    
    # detection_results contain the last detected results for gender and age
    gender_detected = detection_results[-1]['gender'] if detection_results else "Unknown"
    gender_confidence_detected = detection_results[-1]['gender_confidence'] if detection_results else 0.0
    age_detected = detection_results[-1]['age'] if detection_results else "Unknown"
    age_confidence_detected = detection_results[-1]['age_confidence'] if detection_results else 0.0

    modified_huggingface_model = get_modified_model_name(huggingface_model)
    
    log_to_csv(
        logfile_path=logfile_path, 
        image_name=basename, 
        subdir_name=subdir_name, 
        new_file_name=new_file_name, 
        identitynet_strength_ratio=identitynet_strength_ratio, 
        adapter_strength_ratio=adapter_strength_ratio, 
        num_inference_steps=num_inference_steps, 
        guidance_scale=guidance_scale, 
        seed=seed, 
        success=success, 
        error_message=error_message, 
        style_name=style_name, 
        prompt=prompt, 
        negative_prompt=negative_prompt, 
        time_taken=loop_time_taken, 
        current_timestamp=current_timestamp,
        huggingface_model=modified_huggingface_model,
        gender=gender_detected,
        gender_confidence=gender_confidence_detected,
        age=age_detected,
        age_confidence=age_confidence_detected,
        chosen_pose=chosen_pose,
        pose_filename=pose_filename,  
        eye_distance=eye_distance,
        nose_mouth_distance=nose_mouth_distance,
        distance_difference=distance_difference
    )

def log_to_csv(logfile_path, image_name, new_file_name='Unknown', identitynet_strength_ratio=0.0, 
               adapter_strength_ratio=0.0, num_inference_steps=0, guidance_scale=0.0, seed=0, 
               success=True, error_message='', style_name="", prompt="", negative_prompt="", 
               time_taken=0.0, current_timestamp="", huggingface_model="", gender="Unknown", 
               gender_confidence=0.0, age="Unknown", age_confidence=0.0, subdir_name="", chosen_pose="", pose_filename='', eye_distance=0, nose_mouth_distance=0, distance_difference=0): 


    # Ensure the directory exists for the log file
    os.makedirs(os.path.dirname(logfile_path), exist_ok=True)
    # Check if log file exists already
    file_exists = os.path.isfile(logfile_path)

    # Open the CSV file
    with open(logfile_path, 'a', newline='', encoding='utf-8') as csvfile:
        # Define the field names for the CSV
        fieldnames = ['image_name', 'new_file_name', 'identitynet_strength_ratio', 'adapter_strength_ratio', 
                      'num_inference_steps', 'guidance_scale', 'seed', 'success', 'error_message', 'style_name', 
                      'prompt', 'negative_prompt', 'time_taken', 'current_timestamp', 'huggingface_model',
                      'gender', 'gender_confidence', 'age', 'age_confidence','subdir_name','chosen_pose',
                      'most_alike_pose', 'most_alike_eye_distance', 'most_alike_nose_mouth_distance', 'most_alike_distance_difference']  

        # Create a DictWriter object
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        
        # Write the header only if the file is new
        if not file_exists:
            writer.writeheader()
        
        # Write the data row
        writer.writerow({
            'image_name': image_name,
            'new_file_name': new_file_name, 
            'identitynet_strength_ratio': identitynet_strength_ratio,
            'adapter_strength_ratio': adapter_strength_ratio,
            'num_inference_steps': num_inference_steps, 
            'guidance_scale': guidance_scale,
            'seed': seed,
            'success': success,
            'error_message': error_message,
            'style_name': style_name,
            'prompt': prompt,
            'negative_prompt': negative_prompt,
            'time_taken': time_taken,
            'current_timestamp': current_timestamp,
            'huggingface_model': huggingface_model,
            'gender': gender,
            'gender_confidence': gender_confidence,
            'age': age,
            'age_confidence': age_confidence,
            'subdir_name': subdir_name, 
            'chosen_pose': chosen_pose,
            'most_alike_pose': pose_filename,  
            'most_alike_eye_distance': eye_distance,
            'most_alike_nose_mouth_distance': nose_mouth_distance,
            'most_alike_distance_difference': distance_difference
        })

        # Message indicating writing to CSV is complete
        print("\n*******")
        print(f"CONFIGS.PY / log_to_csv def: TADA! Log entry for {image_name} written to {logfile_path}")
        print("\n*******")
########LOG FILE LOGIC DONE############################

########PROCESSING TO PASS TO KUMORI_CLI_ENGINE.PY##############
def list_image_files(input_folder):
    image_patterns = ['**/*.png', '**/*.jpg', '**/*.jpeg']  # Add **/ to search in subdirectories
    files = []
    # Exclude directories named 'needed_padding' from the search
    excluded_dir = 'needed_padding'
    for pattern in image_patterns:
        # Use glob.glob with the recursive parameter to list matching files
        # os.path.join is used to build the path pattern correctly
        pattern_path = os.path.join(input_folder, pattern)
        for file in glob.glob(pattern_path, recursive=True):
            # Check if 'needed_padding' is part of the file's path
            if excluded_dir not in file:
                files.append(file)
    return files

def initial_image(generate_image_func):
    overall_start_time = time.time()
    total_time_taken = 0.0 

    # Call the function to move old logs at the start
    if LOGGING_ENABLED:
        move_old_logs_to_log_folder()

    # Initialize a counter for processed images at the beginning of the function
    processed_images_count = 0

    def get_local_timestamp():
        return datetime.now().strftime("%Y%b%d @ %I:%M%p local time")    

    for loop in range(NUMBER_OF_LOOPS):

        # At the start of each loop iteration, re-query the image files to process
        image_files = list_image_files(INPUT_FOLDER_NAME)
        total_images = len(image_files)  # Get the total number of images to process
        
        # Check if we found any images
        if not image_files:
            raise FileNotFoundError(f"No images found in directory {INPUT_FOLDER_NAME}")
        
        # Print the count of detected image files
        print(f"Processing a total of {len(image_files)} image(s) in '{INPUT_FOLDER_NAME}' at start of loop {loop+1}")
        
        # Shuffle the image files randomly
        random.shuffle(image_files)

        print(f"Starting loop {loop+1} of {NUMBER_OF_LOOPS} at {get_local_timestamp()}")
        print(f"Logging enabled: {LOGGING_ENABLED}")

        for image_number, face_image_path in enumerate(image_files, start=1):
            
            try:
                loop_start_time = datetime.now()
                face_image = [face_image_path]
                basename = os.path.basename(face_image_path)

                # Extract subdirectory name relative to INPUT_FOLDER_NAME
                subdir_name = os.path.relpath(os.path.dirname(face_image_path), INPUT_FOLDER_NAME)
                # Extract subdirectory name relative to INPUT_FOLDER_NAME

                # Check if the file is in the root directory and format subdir_name
                subdir_name = "" if subdir_name == "." else subdir_name

                processed_images_count += 1

                print("\n********************************************************************************")
                print(f"***LOOP {loop + 1} of {NUMBER_OF_LOOPS}: STARTING NEW IMAGE PROCESSING: Image {image_number} of {total_images} in subfolder {subdir_name}***")

                print("********************************************************************************\n")

                # this will compare the incoming face with one in poses.
                directory_to_compare = "poses" 
                verbose_mode = False  # Or True and ESTIMATE_SIMILAR_FACES.PY will print some verbose things about the pose matches

                match_details = compare_faces_main(face_image_path, directory_to_compare, verbose_mode,
                                                GLOBAL_POSE_LIST=GLOBAL_POSE_LIST,
                                                GLOBAL_MALE_POSE_LIST=GLOBAL_MALE_POSE_LIST,
                                                GLOBAL_FEMALE_POSE_LIST=GLOBAL_FEMALE_POSE_LIST)

                # Print the matching file, handling the case where no match was found
                print("\n********************************************************************************")
                if match_details:
                    print("================CONFIGS.PY RECEIVES BACK FROM ESTIMATE_SIMILAR_FACES.PY===================================")

                    print(f"We detected this POSE most closely matched the current image: {match_details['filename']}")
                    print(f"We detected the actual eye distance of: {match_details['eye_distance']}")
                    print(f"We detected the actual nose to mouth distance of: {match_details['nose_mouth_distance']}")
                    print(f"So overall the POSE above has the CLOSEST estimate of scaled overall distance difference of: {match_details['distance_difference']}")
                else:
                    print("ESTIMATE_SIMILAR_FACES.PY detected no POSE closely matched the current image, so sent nothing back, no big whoop, or problem, really.")
                print("\n********************************************************************************")

                #this gets the gender detector going
                detection_results = detect_gender_and_age(face_image_path)

                for result in detection_results:
                    # Print each result individually
                    print("====IF MULTIPLE FACES FOUND, SPLITTING OUT, VIA CONFIGS.PY====")
                    print(f"Singular Gender: {result['gender']}")
                    print(f"Singular Gender Confidence: {result['gender_confidence'] * 100:.2f}%")
                    print(f"Singular Age: {result['age']}")
                    print(f"Singular Age Confidence: {result['age_confidence'] * 100:.2f}%")

                # Initialize variables
                pose_image_path = None
                CHOSEN_PROMPT = MALE_PROMPT

                # Variables for count(s)
                active_anyone_poses = [pose for pose, is_active in GLOBAL_POSE_LIST.items() if is_active]
                active_female_poses = [pose for pose, is_active in GLOBAL_FEMALE_POSE_LIST.items() if is_active]
                active_male_poses = [pose for pose, is_active in GLOBAL_MALE_POSE_LIST.items() if is_active]

                female_with_high_confidence_found = any(result['gender'].lower() == 'female' and result['gender_confidence'] * 100 > GLOBAL_FEMALE_POSE_PERC for result in detection_results)

                if female_with_high_confidence_found:
                    CHOSEN_PROMPT = FEMALE_PROMPT if FEMALE_PROMPT.strip() else MALE_PROMPT
                    if PRESET_POSE_ENABLED and active_female_poses:
                        selected_pose = random.choice(active_female_poses)
                        pose_image_path = [selected_pose]
                        print(f"\n******\nRANDOM_POSE_ENABLED = True\nSelected Pose: {selected_pose} for FEMALE, out of {len(active_female_poses)} poses.")
                        print(f"\n******\nGender match threshold (GLOBAL_FEMALE_POSE_PERC) set at: {GLOBAL_FEMALE_POSE_PERC} \nConfidence score was: {result['gender_confidence'] * 100:.2f}.")

                    else:
                        print("No specific FEMALE pose image available for selection, defaulting to None.")
                        print(f"\n******\nGender match threshold (GLOBAL_FEMALE_POSE_PERC) set at: {GLOBAL_FEMALE_POSE_PERC} \nConfidence in female gender score is: {result['gender_confidence'] * 100:.2f}.")


                elif any(result['gender'].lower() == 'male' and result['gender_confidence'] * 100 > GLOBAL_FEMALE_POSE_PERC for result in detection_results):
                    if PRESET_POSE_ENABLED and active_male_poses:
                        selected_pose = random.choice(active_male_poses)
                        pose_image_path = [selected_pose]
                        print(f"\n******\nRANDOM_POSE_ENABLED = True\nSelected Pose: {selected_pose} for MALE, out of {len(active_male_poses)} poses.")
                    else:
                        print("No specific MALE pose image set for selection, defaulting to None.")

                elif PRESET_POSE_ENABLED and active_anyone_poses:
                    selected_pose = random.choice(active_anyone_poses)
                    pose_image_path = [selected_pose]
                    print(f"\n******\nRANDOM_POSE_ENABLED = True\nSelected Pose: {selected_pose} for ANYONE, out of {len(active_anyone_poses)} poses.")

                print("================================================================================")
                print(f"CONFIGS.PY: Finished setting prompt and pose for {basename}:")
                print(f"PROMPT SELECTED: {CHOSEN_PROMPT}")
                if pose_image_path:
                    print(f"POSE SELECTED: {pose_image_path[0]}")
                else:
                    print("POSE: None selected")
                print("====CONFIGS.PY: POSE AND PROMPT SELECTION COMPLETE====")


                if USE_RANDOM_STYLE:
                    style_name = choose_random_style()
                else:
                    #This is set as an if/then in KUMORI_CLI_ENGINE.PY at around line 300 so must be None if not set by USE_RANDOM_STYLE
                    style_name = None
                
                # Print out the chosen style here
                print(f"CONFIGS.PY:Chosen style for this iteration: {style_name}")

                identitynet_strength_ratio = random.uniform(*IDENTITYNET_STRENGTH_RATIO_RANGE)
                adapter_strength_ratio = random.uniform(*ADAPTER_STRENGTH_RATIO_RANGE)
                num_inference_steps = random.randint(*NUM_INFERENCE_STEPS_RANGE)
                guidance_scale = random.uniform(*GUIDANCE_SCALE_RANGE)
                seed = random.randint(0, 2**32 - 1)


                print("\n===============OVERRIDE CHECKER!=====================")


                print(f"Evaluating final check to set overrides if in filename: {face_image_path}")
                if "female" in face_image_path.lower():
                    selected_pose = choose_random_pose(GLOBAL_FEMALE_POSE_LIST)
                    pose_image_path = [selected_pose] if selected_pose else None

                    CHOSEN_PROMPT = FEMALE_PROMPT
                    print("Force-set gender to FEMALE based on filename; Prompt and pose updated.")
                    print(f"Updated CHOSEN_PROMPT for FEMALE: {CHOSEN_PROMPT}")
                    if selected_pose:
                        print(f"Updated Pose Image Path for FEMALE: {selected_pose}")
                    else:
                        print("No FEMALE pose selected; using generic.")
                elif "male" in face_image_path.lower() and "female" not in face_image_path.lower():
                    selected_pose = choose_random_pose(GLOBAL_MALE_POSE_LIST)
                    pose_image_path = [selected_pose] if selected_pose else None

                    CHOSEN_PROMPT = MALE_PROMPT 
                    print("Force-set gender to MALE based on filename; Prompt and pose updated.")
                    print(f"Updated CHOSEN_PROMPT for MALE: {CHOSEN_PROMPT}")
                    if selected_pose:
                        print(f"Updated Pose Image Path for MALE: {selected_pose}")
                    else:
                        print("No MALE pose selected; using generic.")
                else:
                    print("No explicit gender cue in filename; proceeding with detected gender above!")
                    if pose_image_path:
                        print(f"Previously selected POSE IMAGE PATH: {pose_image_path[0]}")

                # Last chance override or confirmation on PRESET POSE
                if pose_image_path is not None and not PRESET_POSE_ENABLED:
                    print("As a final check, confirming that pose_image_path should be None due to PRESET_POSE_ENABLED being False.")
                    pose_image_path = None

                # Confirm right before usage
                print("CONFIGS.PY: pose_image_path value BEFORE sending to KUMORI_CLI_ENGINE.PY in override call:", pose_image_path)
                
                # New override: Default FEMALE_PROMPT to PROMPT if blank
                if not FEMALE_PROMPT.strip():  # Checks if FEMALE_PROMPT is effectively empty
                    CHOSEN_PROMPT = MALE_PROMPT
                    print("FEMALE_PROMPT is blank, defaulting to MALE_PROMPT for CHOSEN_PROMPT.")

                # Final override check for 'PRESET_POSE_ENABLED'
                if not PRESET_POSE_ENABLED:
                    print("PRESET_POSE_ENABLED is False, setting pose_image_path to None.")
                    pose_image_path = None

                # This ensures the prompt defaults to "a human" if both FEMALE_PROMPT and MALE_PROMPT are blank
                if not FEMALE_PROMPT.strip() and not MALE_PROMPT.strip():
                    CHOSEN_PROMPT = "a human"
                    print("Both FEMALE_PROMPT and MALE_PROMPT are blank, defaulting CHOSEN_PROMPT to \"a human\".")
                else:
                    print(f"CONFIGS.PY: Success in finding gender, therfore CHOSEN_PROMPT sending into KUMORI_CLI_ENGINE.PY with: {CHOSEN_PROMPT}")

                print("==========OVERRIDE CHECKER COMPLETE===============")

                print(f"***LOOP {loop + 1} of {NUMBER_OF_LOOPS}: Processing image {image_number} of {total_images}***")
                print(f"***Filename sending to KUMORI_CLI_ENGINE.PY: {basename}")
                print("================================================================================")
                
                try:
                    # Here, the generate_image_func is called and image processing happens --this is the magic sauce.
                    _, _, generated_file_paths, used_huggingface_model = generate_image_func(
                        face_image=face_image,
                        pose_image=pose_image_path,
                        prompt=CHOSEN_PROMPT,
                        negative_prompt=NEGATIVE_PROMPT,
                        style_name=style_name,
                        enhance_face_region=True,
                        num_steps=num_inference_steps,
                        identitynet_strength_ratio=identitynet_strength_ratio,
                        adapter_strength_ratio=adapter_strength_ratio,
                        guidance_scale=guidance_scale,
                        seed=seed
                    )

                    print(f"Returned HuggingFace Random model from KUMORI_CLI_ENGINE.PY: {used_huggingface_model}")

                    HUGGINGFACE_MODEL = used_huggingface_model

                    # Print settings for the current image BEFORE processing it
                    print_generation_settings(basename, style_name, identitynet_strength_ratio, 
                                            adapter_strength_ratio, num_inference_steps, guidance_scale, seed,
                                            image_number, total_images, HUGGINGFACE_MODEL, CHOSEN_PROMPT)

                except Exception as gen_image_error:
                    print(f"Error during image generation for {face_image_path}: {gen_image_error}. Skipping to next image.")
                    continue  # Safely skip to the next image
                
                loop_end_time = datetime.now()
                loop_time_taken = (loop_end_time - loop_start_time).total_seconds()

                # Immediately print the time taken and current time.
                print(f"Time taken to process image: {loop_time_taken:.2f} seconds")

                # Update the total time taken with this image's processing time
                total_time_taken += loop_time_taken

                # Calculate the average time taken per image
                average_time_per_image = total_time_taken / image_number

                current_timestamp = loop_end_time.strftime("%Y-%m-%d %H:%M:%S")  # Current time after processing
                print(f"Current timestamp: {current_timestamp}")

                # Calculate estimated remaining time considering the images left in this loop and the additional loops
                remaining_images_this_loop = total_images - image_number
                remaining_images_in_additional_loops = (NUMBER_OF_LOOPS - (loop + 1)) * total_images
                total_remaining_images = remaining_images_this_loop + remaining_images_in_additional_loops
                estimated_time_remaining = average_time_per_image * total_remaining_images

                # Display the estimated time remaining including remaining loops
                print(f"Estimated time remaining (including loops): {estimated_time_remaining // 60:.0f} minutes, {estimated_time_remaining % 60:.0f} seconds")

                # Display the overall average time per image in seconds
                print(f"Overall average time per image: {average_time_per_image:.2f} seconds")

                # Display the total number of remaining images to process including looping
                print(f"Total remaining images to process (including loops): {total_remaining_images}")
                print(f"Loop completed at {get_local_timestamp()}")

                # Loop through each generated file
                for generated_file_path in generated_file_paths:
                    input_file_name_without_extension = os.path.splitext(basename)[0]
                    current_timestamp_formatted = datetime.now().strftime("%Y%m%d%H%M%S")

                    # Utilize the get_modified_model_name() function to ensure a clean model name
                    cleaned_model_name = get_modified_model_name(HUGGINGFACE_MODEL)

                    # Convert slashes to underscores for safe filesystem naming
                    model_name_safe = cleaned_model_name.replace("/", "_").replace("\\", "_")

                    # Extract details from match_details or set them to default values if not found
                    pose_filename = match_details.get('filename', '') if match_details else ''
                    eye_distance = match_details.get('eye_distance', 0) if match_details else 0
                    nose_mouth_distance = match_details.get('nose_mouth_distance', 0) if match_details else 0
                    distance_difference = match_details.get('distance_difference', 0) if match_details else 0

                    # Construct the new filename using the sanitized model name
                    new_file_name = f"{model_name_safe}_{input_file_name_without_extension}_{current_timestamp_formatted}.png"
                    new_file_path = os.path.join(OUTPUT_FOLDER_NAME, new_file_name)

                    # Ensuring the output directory exists
                    os.makedirs(OUTPUT_FOLDER_NAME, exist_ok=True)

                    # Rename (move) the file to the new path with the clean filename
                    try:
                        os.rename(generated_file_path, new_file_path)
                        print(f"Image saved as {new_file_path}")
                    except Exception as e:
                        print(f"Error during file renaming/movement to '{new_file_path}': {e}")

                    # Now check if logging is enabled and log the operation
                    if LOGGING_ENABLED:
                        log_image_generation_details(
                            basename=basename,
                            subdir_name=subdir_name,
                            new_file_name=new_file_name,
                            identitynet_strength_ratio=identitynet_strength_ratio,
                            adapter_strength_ratio=adapter_strength_ratio,
                            num_inference_steps=num_inference_steps,
                            guidance_scale=guidance_scale,
                            seed=seed,
                            success=True,
                            error_message="",
                            style_name=style_name,
                            prompt=CHOSEN_PROMPT,
                            negative_prompt=NEGATIVE_PROMPT,
                            loop_time_taken=loop_time_taken,
                            current_timestamp=current_timestamp,
                            huggingface_model=HUGGINGFACE_MODEL,
                            detection_results=detection_results,
                            logfile_path=logfile_path,
                            chosen_pose=selected_pose if selected_pose else "None",
                            pose_filename=pose_filename, 
                            eye_distance=eye_distance,  
                            nose_mouth_distance=nose_mouth_distance,
                            distance_difference=distance_difference
                        )

                print("\n********************************************************************************")
                print(f"*****IMAGE PROCESSING COMPLETE: {new_file_name}*****")
                print("********************************************************************************\n")

                del generated_file_paths  # Explicitly delete large variables
                gc.collect()  # Call garbage collection

            except Exception as gen_image_error:
                print(f"NOPE! Image likely too close up of face. Error during image generation for {face_image_path}: {gen_image_error}. Skipping to next image.")
                print("Trying to add padding, rename, and delete the original image to let it reprocess in the next loop if needed.")         

                # Add padding, rename, and delete the original image
                add_padding_and_rename(face_image_path)
                continue  # Safely skip to the next image
            
    # At the end of the initial_image() function, add:
    total_elapsed_time = time.time() - overall_start_time
    print("\n===FINAL SUMMARY===")
    print(f"Total loops completed: {NUMBER_OF_LOOPS}")
    print(f"Total images processed per loop: {len(image_files)}")
    print(f"Overall total images processed: {NUMBER_OF_LOOPS * len(image_files)}") # Multiplied by the number of loops
    print(f"Overall total time: {total_elapsed_time / 60:.2f} minutes")

########PROCESSING DONE TO PASS TO KUMORI_CLI_ENGINE.PY##############

def add_padding_and_rename(original_image_path, padding_percent=25):
    """
    Adds padding around the image based on a percentage of its dimensions,
    saves it with '_padded' appended to the original name if not already present, 
    and moves the original image to a folder named 'needed_padding'.
    """
    try:
        # Define the needed_padding directory path
        needed_padding_dir = path.join(path.dirname(original_image_path), 'needed_padding')
        # Create the needed_padding directory if it doesn't exist
        makedirs(needed_padding_dir, exist_ok=True)

        with Image.open(original_image_path) as img:
            base_filename, file_extension = path.splitext(original_image_path)
            padded_filename = f"{base_filename}_padded{file_extension}"

            # Check if "_padded" is already in the filename to prevent duplicate naming
            if "_padded" in base_filename:
                print(f"Already padded, going to need to find another way to fix it, TBD. Check image.")
                # Move the already padded image to the needed_padding directory
                move(original_image_path, path.join(needed_padding_dir, path.basename(original_image_path)))
                print(f"Moved already padded image to '{needed_padding_dir}'")
                return

            width, height = img.size
            padding_width = int(width * padding_percent / 100)
            padding_height = int(height * padding_percent / 100)

            new_width = width + 2 * padding_width
            new_height = height + 2 * padding_height

            # Create a new image with the padded size and a white background
            new_img = Image.new('RGB', (new_width, new_height), "white")
            new_img.paste(img, (padding_width, padding_height))
            
            # Save the new (padded) image and keep the original name if "_padded" is already there
            new_img.save(padded_filename)
            print(f"Padded image saved as: {padded_filename}")

    except Exception as e:
        print(f"Error adding padding to {original_image_path}: {e}")
    
    # Move the original image to the 'needed_padding' directory if a new padded file was created
    if padded_filename != original_image_path:
        try:
            move(original_image_path, path.join(needed_padding_dir, path.basename(original_image_path)))
            print(f"Moved original image to '{needed_padding_dir}'.")
        except Exception as e:
            print(f"Error moving original image '{original_image_path}' to '{needed_padding_dir}': {e}")


# def print_generation_settings(basename, style_name, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, image_number, total_images, HUGGINGFACE_MODEL):  

def print_generation_settings(basename, style_name, identitynet_strength_ratio, adapter_strength_ratio, num_inference_steps, guidance_scale, seed, image_number, total_images, HUGGINGFACE_MODEL, CHOSEN_PROMPT):


    print("===IMAGE GENERATION DATA SUMMARY===")    
    # Existing print statements follow
    print(f"- Image {image_number} of {total_images}\n"
          f"- Filename: {basename}\n"
          f"- Use random style: {USE_RANDOM_STYLE}\n"
          f"- Style: {style_name}\n"
          f"- IdentityNet strength ratio: {identitynet_strength_ratio:0.2f}\n"
          f"- Adapter strength ratio: {adapter_strength_ratio:0.2f}\n"
          f"- Number of inference steps: {num_inference_steps}\n"
          f"- Guidance scale: {guidance_scale:0.2f}\n"
          f"- Seed: {seed}\n"
          f"- Input folder name: {INPUT_FOLDER_NAME}\n"
          f"- Output folder name: {OUTPUT_FOLDER_NAME}\n"
          f"- Prompt: {CHOSEN_PROMPT}\n"
          f"- Negative prompt: {NEGATIVE_PROMPT}\n"
          f"- Number of loops: {NUMBER_OF_LOOPS}\n"
          f"- HuggingFace Model: {HUGGINGFACE_MODEL}\n")

    print("===DEFINING COMPLETE, GENERATING IMAGE...===")
import cv2
import os
import dlib
import numpy as np

VERBOSE_MODE = False

MODEL_DIR = "facial_landmarks_model"
INPUT_DIRECTORY = 'poses'
TEST_PERSON_IMAGE = 'test_person.jpeg'
LANDMARKS_MODEL_PATH = os.path.join(MODEL_DIR, "shape_predictor_68_face_landmarks.dat")

class FaceEvaluator:
    def __init__(self, input_dir, test_image_path, landmarks_model_path, verbose=True, **kwargs):
        self.input_dir = input_dir
        self.test_image_path = test_image_path
        self.landmarks_detector = dlib.shape_predictor(landmarks_model_path)
        self.face_detector = dlib.get_frontal_face_detector()
        self.verbose = verbose  
        

        # Store the global pose lists directly in the instance
        self.global_pose_list = kwargs.get("GLOBAL_POSE_LIST", {})
        self.global_male_pose_list = kwargs.get("GLOBAL_MALE_POSE_LIST", {})
        self.global_female_pose_list = kwargs.get("GLOBAL_FEMALE_POSE_LIST", {})


        self.valid_extensions = {'.jpeg', '.jpg', '.png','webp'}
    
    def get_landmarks(self, image, rectangle):
        return np.matrix([[p.x, p.y] for p in self.landmarks_detector(image, rectangle).parts()])
    
    def get_nose_to_mouth_distance(self, landmarks):
        nose_tip = landmarks[33]  # Tip of the nose
        mouth_center = np.mean(landmarks[48:68], axis=0)  # Average of the mouth landmarks
        distance = np.linalg.norm(nose_tip - mouth_center)
        return distance

    def get_eye_distance(self, landmarks):
        right_eye = landmarks[36:42]
        right_eye_center = np.mean(right_eye, axis=0)
        left_eye = landmarks[42:48]
        left_eye_center = np.mean(left_eye, axis=0)
        eye_distance = np.linalg.norm(right_eye_center - left_eye_center)
        return eye_distance

    def evaluate_test_image(self):
        file_name = os.path.basename(self.test_image_path)
        file_size = os.path.getsize(self.test_image_path)
        image = cv2.imread(self.test_image_path)
        if image is None:
            print(f"Could not read the test image {file_name}.")
            return None
        image_shape = image.shape
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector(gray, 1)
        if not faces:
            print("No faces detected in the test image.")
            return None
        test_face_landmarks = self.get_landmarks(gray, faces[0])
        test_eye_distance = self.get_eye_distance(test_face_landmarks)
        test_nose_mouth_distance = self.get_nose_to_mouth_distance(test_face_landmarks)
        print("==================ESTIMATE_SIMILAR_FACES.PY BEGIN PROCESSING=====================================")
        print("\nCurrent incoming image details:")
        print(f"File Name: {file_name}")
        print(f"File Size: {file_size} bytes")
        print(f"Image Shape (Height, Width, Channels): {image_shape}")
        print(f"Curent Image Eye Distance: {test_eye_distance}")
        print(f"Curent Image Nose to Mouth Center Distance: {test_nose_mouth_distance}")
        return test_eye_distance, test_nose_mouth_distance


    def find_closest_match(self):
        test_metrics = self.evaluate_test_image()
        if test_metrics is None:
            print("ESTIMATE_SIMILAR_FACES.PY: No face detected in the provided incoming image.")
            return None

        test_eye_distance, test_nose_mouth_distance = test_metrics

        test_image = cv2.imread(self.test_image_path)
        test_image_width = test_image.shape[1]
        print("ESTIMATE_SIMILAR_FACES.PY: ...reading poses.")

        if self.verbose:
            print("====COMPARISON RESULTS====")
        
        results = []

        for file_name in os.listdir(self.input_dir):
            file_path = os.path.join(self.input_dir, file_name).replace("\\", "/")  # Ensure consistency in path format

            if file_name.lower().endswith(tuple(self.valid_extensions)):
                # Check if file is flagged as True in any of the global lists
                if self.global_pose_list.get(file_path) or self.global_male_pose_list.get(file_path) or self.global_female_pose_list.get(file_path):
                    image_path = os.path.join(self.input_dir, file_name)
                    file_size = os.path.getsize(image_path)
                    image = cv2.imread(image_path)
                    if image is None:
                        print(f"Could not read the image {file_name}, skipping...")
                        continue

                    image_height, image_width, _ = image.shape
                    width_ratio = image_width / test_image_width

                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    faces = self.face_detector(gray, 1)
                    if not faces:
                        # print("++++++++++")
                        # print(f"ESTIMATE_SIMILAR_FACES.PY: No faces detected in the image: {file_name}. Skipping...")
                        # print("++++++++++")
                        continue

                    face_landmarks = self.get_landmarks(gray, faces[0])
                    eye_distance = self.get_eye_distance(face_landmarks)
                    nose_mouth_distance = self.get_nose_to_mouth_distance(face_landmarks)
                    scaled_eye_distance = eye_distance / width_ratio
                    scaled_nose_mouth_distance = nose_mouth_distance / width_ratio

                    distance_difference = abs(scaled_eye_distance - test_eye_distance) + abs(scaled_nose_mouth_distance - test_nose_mouth_distance)
                    
                    results.append((file_name, file_size, (image_height, image_width), eye_distance, nose_mouth_distance, distance_difference, scaled_eye_distance, scaled_nose_mouth_distance))

                    if self.verbose:
                        print("++++++++++")
                        print(f"File Name: {file_name}")
                        print(f"File Size: {file_size} bytes")
                        print(f"Image Shape (Height, Width, Channels): ({image_height}, {image_width}, 3)")
                        print(f"Eye Distance: {eye_distance}")
                        print(f"Nose to Mouth Center Distance: {nose_mouth_distance}")
                        print(f"Overall Distance Difference from Test Image: {distance_difference}")
                        print(f"Estimated Scaled Eye Distance: {scaled_eye_distance}")
                        print(f"Estimated Scaled Nose to Mouth Distance: {scaled_nose_mouth_distance}")
                        print("++++++++++")
        
        # After processing all files, find the best match or report no match.
        if results:
            best_estimated_match = min(results, key=lambda x: x[5])
            print("\n====BEST ESTIMATED MATCHING POSE====")
            print(f"The closest matching estimated image is: {best_estimated_match[0]}")
            print(f"File Size: {best_estimated_match[1]} bytes")
            print(f"Image Shape (Height, Width, Channels): {best_estimated_match[2]}")
            print(f"With an actual eye distance of: {best_estimated_match[3]}")
            print(f"And an actual nose to mouth distance of: {best_estimated_match[4]}")
            print(f"Estimated scaled overall distance difference of: {best_estimated_match[5]}")
            
            # Print the summary only if verbose mode is True
            if self.verbose:
                self.print_summary()

            print("==================ESTIMATE_SIMILAR_FACES.PY ENDS PROCESSING=====================================")
            
            # Preparing a detailed response including necessary match details
            match_details = {
                'filename': best_estimated_match[0],
                'eye_distance': best_estimated_match[3],
                'nose_mouth_distance': best_estimated_match[4],
                'distance_difference': best_estimated_match[5]
            }
            return match_details
        else:
            print("ESTIMATE_SIMILAR_FACES.PY: No matching image found based on estimated scale.")
            return None  # or an appropriate message

def print_summary(self):
    print("\n====SUMMARY====")
    print("The comparisons are made based on pixel differences in facial landmarks (eye and nose to mouth distances), ")
    print("normalized by the width ratio between each comparison image and the test image. This normalization assists in ")
    print("accounting for discrepancies in image sizes, ensuring a fairer comparison. The 'Estimated Scaled Overall Distance Difference' ")
    print("is a calculated measure of similarity, taking into account the scaling factor due to image size differences, where a smaller value ")
    print("indicates a closer match. The closest matching image is determined by finding the minimum of these scaled difference values, ")
    print("representing the most similar features to those of the test image within the given dataset.\n")
    
    print("The rationale behind this approach is rooted in the idea that facial proportions should remain consistent across images ")
    print("of different sizes. By scaling the distances of facial landmarks relative to the image dimensions, we aim to compare ")
    print("these proportions rather than absolute measurements. Thus, the identified 'best match' is the image whose facial feature proportions ")
    print("most closely mimic those of the test image, despite any discrepancies in image resolution or size.")

def main(test_image_path, input_directory, verbose=True, **kwargs):
    # Create an evaluator instance, now passing kwargs as well
    evaluator = FaceEvaluator(input_directory, test_image_path, LANDMARKS_MODEL_PATH, verbose, **kwargs)
  
    # Access and print the additional arguments if they exist
    # if "GLOBAL_POSE_LIST" in kwargs:
    #     print("Global Pose List:", kwargs["GLOBAL_POSE_LIST"])
    # if "GLOBAL_MALE_POSE_LIST" in kwargs:
    #     print("Global Male Pose List:", kwargs["GLOBAL_MALE_POSE_LIST"])
    # if "GLOBAL_FEMALE_POSE_LIST" in kwargs:
    #     print("Global Female Pose List:", kwargs["GLOBAL_FEMALE_POSE_LIST"])
    
    # This now returns the closest match filename or None
    closest_match_filename = evaluator.find_closest_match()
    
    # Return this filename to the caller
    return closest_match_filename

if __name__ == "__main__":
    import sys

    if len(sys.argv) == 3:
        test_image = sys.argv[1]
        directory = sys.argv[2]
    else:
        test_image = TEST_PERSON_IMAGE
        directory = INPUT_DIRECTORY

    matching_file = main(test_image, directory, VERBOSE_MODE)
    if matching_file:
        print(f"ESTIMATE_SIMILAR_FACES.PY: The file match is: {matching_file}")
    else:
        print("ESTIMATE_SIMILAR_FACES.PY: No matching file found.")
import cv2
import os

# Model file paths
faceProto = "gender_detect/opencv_face_detector.pbtxt"
faceModel = "gender_detect/opencv_face_detector_uint8.pb"
ageProto = "gender_detect/age_deploy.prototxt"
ageModel = "gender_detect/age_net.caffemodel"
genderProto = "gender_detect/gender_deploy.prototxt"
genderModel = "gender_detect/gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)
    net.setInput(blob)
    detections = net.forward()
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3] * frameWidth)
            y1 = int(detections[0,0,i,4] * frameHeight)
            x2 = int(detections[0,0,i,5] * frameWidth)
            y2 = int(detections[0,0,i,6] * frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8) 
    return frameOpencvDnn, faceBoxes

def detect_gender_and_age(image_path):
    frame = cv2.imread(image_path)
    if frame is None:
        print(f"Skipping {image_path}, failed to load.")
        return {}

    print("====GENDER DETECTION IMAGE START VIA GENDER_DETECT.PY====")
    print(f"Image name: {image_path.split('/')[-1]}")  # Display image name

    _, faceBoxes = highlightFace(faceNet, frame)

    results = []  # Initialize an empty list to store results from potentially multiple faces

    padding = 20
    _, faceBoxes = highlightFace(faceNet, frame)

    if len(faceBoxes) == 0:
        print("No face detected")
    else:
        print(f"Number of faces detected: {len(faceBoxes)}")

    for faceBox in faceBoxes:
        face = frame[max(0, faceBox[1]-padding):min(faceBox[3]+padding,frame.shape[0]-1),
                     max(0, faceBox[0]-padding):min(faceBox[2]+padding,frame.shape[1]-1)]
        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)

        # Gender Prediction
        genderNet.setInput(blob)
        genderPreds = genderNet.forward()
        genderClass = genderPreds[0].argmax()
        gender = genderList[genderClass]
        genderConfidence = genderPreds[0][genderClass]

        # Age Prediction
        ageNet.setInput(blob)
        agePreds = ageNet.forward()
        ageClass = agePreds[0].argmax()
        age = ageList[ageClass]
        ageConfidence = agePreds[0][ageClass]

        print(f"Gender: {gender}")
        print(f"Gender confidence: {genderConfidence * 100:.2f}%")
        print(f"Age: {age} years")
        print(f"Age confidence: {ageConfidence * 100:.2f}%")

        # Append the result for the current face to the list
        results.append({
            'gender': gender,
            'gender_confidence': genderConfidence,
            'age': age,
            'age_confidence': ageConfidence
        })

    print("====GENDER DETECTION IMAGE END====\n")

    # Return a list of dictionaries (one dictionary per detected face)
    return results

# Optionally, keep a simple test when module itself is run
if __name__ == "__main__":
    test_image = input("Enter test image path: ")
    detect_gender_and_age(test_image)
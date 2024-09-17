import tensorflow as tf
import cv2
import numpy as np
import os

#load the object recognition model
model = tf.saved_model.load("/Users/blakeweiss/Desktop/objectrecognitionARGUS/pretrainedmodelobj")

#path to known faces folder
known_faces_folder = "/Users/blakeweiss/Desktop/NewOBJrecogsystemfacerecog/knownfaces"

#load known faces from folder
def load_known_faces():
    known_faces = {}
    for filename in os.listdir(known_faces_folder):
        if filename.endswith(".jpg") or filename.endswith(".png"):
            img_path = os.path.join(known_faces_folder, filename)
            name = os.path.splitext(filename)[0]
            img = cv2.imread(img_path)
            known_faces[name] = img
    return known_faces

#detect faces and compare with known faces
def detect_and_recognize_faces(image, known_faces):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    recognized_faces = []

    for (x, y, w, h) in faces:
        face_roi = image[y:y+h, x:x+w]
        best_match_name = "Unknown"
        min_distance = float('inf')

        for name, known_face in known_faces.items():
            #resize the known face to match the detected face ROI
            known_face_resized = cv2.resize(known_face, (w, h))

            #compare the faces using norm
            diff = cv2.norm(face_roi, known_face_resized, cv2.NORM_L2)
            if diff < min_distance:
                min_distance = diff
                best_match_name = name

        recognized_faces.append((x, y, w, h, best_match_name))
    
    return recognized_faces

def detect_objects(image, confidence_threshold=0.5):
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]
    detections = model(input_tensor)

    #detections to numpy arrays
    num_detections = int(detections.pop('num_detections'))
    detections = {key: value[0, :num_detections].numpy() for key, value in detections.items()}

    #confidence threshold filter
    detection_scores = detections['detection_scores']
    mask = detection_scores >= confidence_threshold

    #filter out low confidence detections
    for key in detections:
        detections[key] = detections[key][mask]

    detections['num_detections'] = len(detections['detection_scores'])
    detections['detection_classes'] = detections['detection_classes'].astype(np.int64)

    return detections

def objectrecognitionrun():
    #label map
    label_map = {
        1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
        7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant",
        13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
        19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra",
        25: "giraffe", 27: "backpack", 28: "umbrella", 31: "handbag", 32: "tie", 33: "suitcase",
        34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite", 39: "baseball bat",
        40: "baseball glove", 41: "skateboard", 42: "surfboard", 43: "tennis racket", 44: "bottle",
        46: "wine glass", 47: "cup", 48: "fork", 49: "knife", 50: "spoon", 51: "bowl", 52: "banana",
        53: "apple", 54: "sandwich", 55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog",
        59: "pizza", 60: "donut", 61: "cake", 62: "chair", 63: "couch", 64: "potted plant",
        65: "bed", 67: "dining table", 70: "toilet", 72: "tv", 73: "laptop", 74: "mouse",
        75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven", 80: "toaster",
        81: "sink", 82: "refrigerator", 84: "book", 85: "clock", 86: "vase", 87: "scissors",
        88: "teddy bear", 89: "hair drier", 90: "toothbrush"
    }

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Error: Camera not accessible or not found.")
        return

    detected_labels = set()
    known_faces = load_known_faces()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame")
            break

        #object detection
        detections = detect_objects(frame)

        #detection classes to labels
        detection_labels = [label_map.get(class_id, 'Unknown') for class_id in detections['detection_classes']]

        for i, label in enumerate(detection_labels):
            confidence = detections['detection_scores'][i]
            detected_labels.add(label)

            #bounding box for detected objects
            bbox = detections['detection_boxes'][i]
            ymin, xmin, ymax, xmax = bbox
            (left, right, top, bottom) = (xmin * frame.shape[1], xmax * frame.shape[1],
                                          ymin * frame.shape[0], ymax * frame.shape[0])
            cv2.rectangle(frame, (int(left), int(top)), (int(right), int(bottom)), (255, 0, 0), 2)
            boxlabel = f'{label}: {int(confidence * 100)}%'
            cv2.putText(frame, boxlabel, (int(left), int(top) - 10), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)

        #face detection and recognition
        recognized_faces = detect_and_recognize_faces(frame, known_faces)

        for (x, y, w, h, name) in recognized_faces:
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)
            cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
            cv2.putText(frame, name, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2)

        #display the result
        cv2.imshow('ARGUS object and face recognition', frame)

        #'q' to quit
        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    print(list(detected_labels))

objectrecognitionrun()

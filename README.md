ARGUS Object Recognition Component
----------

Overview
----------
This is an object recognition system designed as a component of the ARGUS project. It uses TensorFlow for real-time object detection via your computer’s webcam. The system is capable of identifying a variety of objects and displaying bounding boxes around them along with confidence scores.

Prerequisites
----------
Ensure the following dependencies are installed:

	•	Python 3.7+
	•	TensorFlow 2.x
	•	OpenCV
	•	NumPy

You can install the dependencies using the following command:

pip3.10 install tensorflow opencv-python-headless numpy

Setup
----------
	1.	Place your TensorFlow model in the specified directory: /Users/blakeweiss/Desktop/objectrecognitionARGUS/pretrainedmodelobj/. If your model is stored elsewhere, update the path in the script accordingly.
	2.	Connect a webcam to your computer for real-time object detection.

Running the Object Recognition
----------
To run the object recognition system, execute the script:

python3.10 your_script_name.py

The program will capture video from the webcam for a duration of 10 seconds by default (adjustable by modifying the capture_duration parameter). During this time, a window will display the detected objects, their bounding boxes, and confidence levels.

Press q to exit the application before the capture duration ends.

Output
----------
At the end of the capture duration, the program prints a list of unique objects detected during the session.

Customization
----------
	•	Confidence Threshold: Modify the detection confidence threshold by changing the confidence > 0.5 condition within the objectrecognitionrun function to a different value.
	•	Label Map: The label map dictionary links numerical class IDs to their respective object names. Adjust this dictionary as needed to align with the labels provided by your TensorFlow model.

Troubleshooting
----------
	•	Ensure that the TensorFlow model is correctly loaded and the video feed is functional if no objects are detected.
	•	Verify that all dependencies are installed and properly configured.
	•	If the webcam does not work, try changing the device index in cv2.VideoCapture(0) to another index if multiple cameras are available.

License
----------
This component is licensed under the MIT License.

Feel free to adjust this README to better fit your project’s needs or specific details.
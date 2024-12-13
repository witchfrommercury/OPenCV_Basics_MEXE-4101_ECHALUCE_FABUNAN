# Echaluce_Fabunan_DrawingRectanglesOnCarsOnTheHighway

### Drawing Rectangles On Cars On The Highway

### 🚗 Introduction

- In conclusion, geometric shapes are very important for computer vision tasks that have to do with cars; they are the critical tool for researchers and practitioners. They give a visual and mathematical representation of vehicles in a way that closes the gap between raw visual data and actionable insights. It is either in bounding boxes, polygons, or keypoints, which help algorithms to understand, process, and analyze complex scenes involving cars.

- Their applications range from far and wide, driving innovations in high-tech fields such as autonomous driving, where accurate car detection and tracking are essential for safety and efficiency. In traffic management, these shapes help monitor vehicle movements, analyze congestion patterns, and enforce regulations. Similarly, in smart surveillance systems, they enhance the ability to identify vehicles, recognize license plates, and even detect suspicious activities.

- Geometric shapes also find an important place in the improvement of model interpretability, validation of predictions, and rigorous testing. These shapes give a tangible means of visualizing model outputs, thus pointing out errors or areas for improvement. By using them in data augmentation and edge case handling, these shapes contribute to making robust and versatile models capable of performing reliably in various real-world scenarios.Therefore, the use of geometric shapes in computer vision not only enhances understanding of visual data but also propels innovation across a multitude of sectors, thus making them essential in the pursuit of smarter, safer, and more efficient technological solutions.

### 🚗 Abstract

- Develop Vehicle Detection Models Accurately
  
Implement computer vision algorithms based on geometric shapes, including bounding boxes and polygons, for the robust detection and localization of vehicles in a variety of settings and conditions.

- Improve feature recognition and segmentation
  
Make use of geometric annotations to enhance key vehicle feature extraction and analysis for accurate segmentation that is used in applications like damage assessment and vehicle classification.

- Allow Real-Time Tracking and Analysis
  
Design algorithms that can apply geometric shapes towards real-time vehicle movement tracking for autonomous driving and support for management systems in traffic.

- Annotated Vehicle Dataset
  
The data comprises fully annotated vehicle images through geometrical shapes bounding box polygons keypoints suitable for any applications: detection, segmentation tracking.

- Trained Computer Vision Models
  
Highly performing machine learning models trained to identify, segment, and analyze vehicles under a variety of scenarios with a focus on occlusions and low-visibility conditions.

### 🚗 Project Method

- Import Libraries
```   
import numpy as np 
import cv2
```
• numpy: A library for numerical computations. Although it's imported here, it is not directly used 
in the provided code. 
• cv2: OpenCV, a library for computer vision tasks such as image processing, video analysis, and 
object detection.

---

- Define File Paths
```   
haar_cascade = 'haarcascade_car.xml'  # Path to Haar Cascade file 
video = 'cars5.mp4'  # Path to the video file 
```
• haar_cascade: Specifies the path to the Haar Cascade XML file used for car detection. This file 
contains pre-trained data for detecting cars in images. 
• video: Specifies the path to the input video file (cars5.mp4).

---
- Open the Video File
```
cap = cv2.VideoCapture(video) 
car_cascade = cv2.CascadeClassifier(haar_cascade)
```
• cv2.VideoCapture(video): Opens the video file for processing. The cap object is used to read 
video frames. 
• cv2.CascadeClassifier(haar_cascade): Loads the Haar Cascade XML file for detecting cars. The 
car_cascade object is used to perform detection.

---
- Retrieve Video Properties
```
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
fps = int(cap.get(cv2.CAP_PROP_FPS))
```
• cv2.CAP_PROP_FRAME_WIDTH: Retrieves the width of each video frame. 
• cv2.CAP_PROP_FRAME_HEIGHT: Retrieves the height of each video frame. 
• cv2.CAP_PROP_FPS: Retrieves the video's frames per second (FPS). 
These properties are used to configure the output video.

---
- Define Video Writer
```
fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Codec for mp4 format 
out = cv2.VideoWriter('Car_Detector5.mp4', fourcc, fps, (frame_width, frame_height))
```
• cv2.VideoWriter_fourcc(*'mp4v'): Specifies the codec for encoding the output video. 'mp4v' is 
used for MP4 files. 
• cv2.VideoWriter(): Creates a VideoWriter object out to save the processed video. Parameters:  
o 'Car_Detector5.mp4': Output file name. 
o fourcc: Codec for encoding. 
o fps: Frames per second for the output video. 
o (frame_width, frame_height): Frame dimensions of the output video.

---
- Process Video Frames
```
while cap.isOpened(): 
ret, frame = cap.read()
if not ret: 
break
```
• cap.isOpened(): Checks if the video file is open and ready for processing. 
• cap.read(): Reads the next frame from the video.  
o ret: Boolean indicating if the frame was read successfully. 
o frame: The current frame. 
If no frame is read (ret is False), the loop exits.

---
- Convert Frame to Grayscale
```
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
```
• cv2.cvtColor(): Converts the frame from color (BGR) to grayscale. Haar Cascade works better on 
grayscale images because it reduces computational complexity.

---
- Detect Cars
```
cars = car_cascade.detectMultiScale(gray, 1.3, 5)
```
• detectMultiScale(): Detects objects (cars) in the grayscale image. Parameters:  
o gray: Input grayscale frame. 
o 1.3: Scale factor for resizing the image during detection. Larger values make detection 
faster but less accurate. 
o 5: Minimum number of neighbors for a rectangle to be considered a valid detection. 
The function returns a list of bounding boxes for detected cars, where each bounding box is represented 
as (x, y, w, h): 
• (x, y): Top-left corner of the bounding box. 
• (w, h): Width and height of the bounding box.

---
- Draw Bounding Boxes
```
for (x, y, w, h) in cars: 
cv2.rectangle(frame, (x, y), (x + w + 5, y + h + 5), (0, 0, 255), 3)
```
• cv2.rectangle(): Draws rectangles around detected cars on the original frame. Parameters:  
o frame: The original frame to draw on. 
o (x, y): Top-left corner of the rectangle. 
o (x + w + 5, y + h + 5): Bottom-right corner (slightly extended for better visibility). 
o (0, 0, 255): Rectangle color (red in BGR format). 
o 3: Thickness of the rectangle.

---
- Write Processed Frame
```
out.write(frame)
```
• Writes the processed frame (with bounding boxes) to the output video file.

---
- Release Resources
```
cap.release() 
out.release()
```
• cap.release(): Closes the video file and releases resources associated with reading it. 
• out.release(): Closes the output video file and releases resources associated with writing it. 
This ensures proper cleanup after processing.

---
Overall Workflow 
1. Open the video file and load the Haar Cascade model. 
2. Read frames from the video in a loop. 
3. Convert each frame to grayscale and detect cars using the Haar Cascade. 
4. Draw bounding boxes around detected cars. 
5. Save the processed frames to a new video file. 
6. Release resources when processing is complete.

### 🚗 Conclusion

- This program is used to detect cars in a video using a Haar Cascade classifier as the feature descriptor. It takes frames from a given video file, detects cars, draws rectangles around them, and then writes the video output. In testing the system on five videos, it was observed that for some of the videos the system was only able to identify one car per frame, while for other videos the system was able to identify multiple cars in the same frame. This variability may be attributed to the quality of the videos used, lighting, camera view point, or just the nature of the Haar Cascade classifier. There could also be false negatives and inconsistent detections due to the use of pre-trained Haar Cascade features in the system (haarcascade_car.xml). Despite the fact that the code does its job of detecting faces in videos successfully, there are built-in object detection models like YOLO and SSD that could be used, or detection parameters could be tweaked for better results, or the quality of the videos could be enhanced using some pre-processing techniques. In summary, the system presented herein exhibits functional performance in car detection, but there is certainly room for improvement in terms of accuracy and stability.

### 🚗 Additional Materials

[] Drawing Rectangles On Cars On The Highway

![image](https://github.com/user-attachments/assets/77e6fa78-0643-4fb7-a9cb-e69c8bfab0e3)

---

![image](https://github.com/user-attachments/assets/bac81746-5142-4d18-99b3-fbec35d0f53d)

![image](https://github.com/user-attachments/assets/2ee39d3d-978a-411f-b2bc-e4eef5485554)

![image](https://github.com/user-attachments/assets/be388145-0dbb-4a5d-ab50-06e262c4e431)

![image](https://github.com/user-attachments/assets/b84a2b70-0291-4999-8847-81a413afbc77)

![image](https://github.com/user-attachments/assets/92328f6a-f9ff-49dd-bae1-fe9bc259d0ec)

---

### References

https://github.com/misbah4064/car_detector_haarcascades

https://www.youtube.com/watch?v=zc6AP7B-CgI








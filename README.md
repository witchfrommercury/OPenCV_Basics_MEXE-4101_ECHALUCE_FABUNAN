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


- Import Necessary Libraries
  
Use libraries like OpenCV and NumPy for image and video processing.

- Load Pre-trained Model
  
Load a Haar Cascade XML file (or other detection models, such as YOLO or SSD) for vehicle detection.

- Video Input
  
Read the video file or capture video frames from a live feed.

- Preprocess Video Frames
  
Convert each frame to grayscale to improve detection accuracy.
Resize or normalize frames if necessary to fit model requirements.

- Vehicle Detection
 
Apply the detection model (in this case, the Haar Cascade) to classify the objects in the frame.
Retrieve the bounding boxes for the vehicles

- Draw Geometric Shapes

Overlay rectangles or other geometric shapes around the vehicles that are classified

- Save the Output Processed
  
Write the output into a new video file, or stream them out in real time


### 🚗 Conclusion

- This program is used to detect cars in a video using a Haar Cascade classifier as the feature descriptor. It takes frames from a given video file, detects cars, draws rectangles around them, and then writes the video output. In testing the system on five videos, it was observed that for some of the videos the system was only able to identify one car per frame, while for other videos the system was able to identify multiple cars in the same frame. This variability may be attributed to the quality of the videos used, lighting, camera view point, or just the nature of the Haar Cascade classifier. There could also be false negatives and inconsistent detections due to the use of pre-trained Haar Cascade features in the system (haarcascade_car.xml). Despite the fact that the code does its job of detecting faces in videos successfully, there are built-in object detection models like YOLO and SSD that could be used, or detection parameters could be tweaked for better results, or the quality of the videos could be enhanced using some pre-processing techniques. In summary, the system presented herein exhibits functional performance in car detection, but there is certainly room for improvement in terms of accuracy and stability.

### 🚗 Additional Materials

![image](https://github.com/user-attachments/assets/77e6fa78-0643-4fb7-a9cb-e69c8bfab0e3)

![image](https://github.com/user-attachments/assets/bac81746-5142-4d18-99b3-fbec35d0f53d)

![image](https://github.com/user-attachments/assets/2ee39d3d-978a-411f-b2bc-e4eef5485554)

![image](https://github.com/user-attachments/assets/be388145-0dbb-4a5d-ab50-06e262c4e431)

![image](https://github.com/user-attachments/assets/b84a2b70-0291-4999-8847-81a413afbc77)

![image](https://github.com/user-attachments/assets/92328f6a-f9ff-49dd-bae1-fe9bc259d0ec)







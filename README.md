<h1 align="center">Extracting Contours for Shape Analysis in Car Images</h1>
<h2 align="center">Final Requirement: Basic Open CV</h2>
<h3 align="center">MExEE 402 - MExE Elective 2</h3>
<br>

## Table of Contents
  - [I. Abstract](#i-abstract)
  - [II. Introduction](#ii-introduction)
  - [III. Project Methods](#iii-project-methods)
  - [IV. Conclusion](#iv-conclusion)
  - [V. Additional Materials](#v-additional-materials)
  - [VI. References](#vi-references)
  - [VII. Group Members](#vii-group-members)
<hr> 
<br>


## I. Abstract

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This project focuses on extracting contours from car images and analyzing their geometric features using OpenCV. Contour detection is a critical technique in computer vision that enables precise identification and analysis of object boundaries. By leveraging edge detection and contour extraction methods, the project addresses key challenges such as noisy backgrounds, inconsistent lighting, and varying image quality in car datasets.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The project approach involves preprocessing car images to enhance clarity, applying advanced contour detection algorithms to isolate object shapes, and performing detailed shape analysis to extract valuable geometric features. These features include properties such as perimeter, area, and bounding rectangles, which can be used in applications like vehicle recognition, design evaluation, and structural analysis.

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; The expected results include improved preprocessing pipelines to handle image noise and inconsistencies, highly accurate contour identification for isolating car shapes, and a comprehensive understanding of vehicle geometries through advanced visualization techniques.
<br>
<br>


## II. Introduction
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; This project is part of the MeXE 402 - Mechatronics Engineering Elective 2: Data Science and Machine Learning course final exam.
- **Goal of the Project:** The primary goal of this project is to extract contours from car images and perform a detailed analysis of the identified shapes using OpenCV tools. This process not only improves understanding of image processing techniques but also provides practical applications in real-world scenarios, such as vehicle recognition and structural analysis.
- **Overview of OpenCV:**
    - OpenCV (Open Source Computer Vision Library) is an open-source library of programming functions primarily aimed at real-time computer vision tasks.
    - It provides a comprehensive suite of tools for image and video analysis, including object detection, image segmentation, and feature extraction.
- **Significance of Shape Analysis**
    - Shape analysis is a fundamental problem in computer vision, with diverse applications such as object recognition, image segmentation, and graphics processing.
    - Understanding shapes enables machines to interpret visual data effectively, making it crucial in areas like autonomous systems and digital graphics.
- **Contour Detection in Car Images**
    - Shape analysis plays a critical role in computer vision, enabling the understanding and classification of objects based on their geometric features.
    - Contour extraction is a fundamental technique used to identify and analyze object boundaries with precision.
    - This project specifically focuses on car images, where detecting and analyzing contours can provide detailed insights into the unique shapes, dimensions, and structural features of vehicles.
    - Contour detection allows for applications such as vehicle recognition, shape classification, and segmentation, which are essential in fields like automated driving systems, traffic management, and design analysis.
    - By leveraging OpenCV's capabilities, this project highlights the importance of effective pre-processing and detection techniques in overcoming challenges such as noisy backgrounds and varied lighting conditions.

<br>
<br>


## III. Project Methods
- **Dataset Preparation:**
    - Collect a dataset of car images from reliable online sources.
    - Preprocess the images by resizing, normalizing, and converting them to grayscale for consistency.
- **Contour Detection:**
    - Apply edge detection methods, such as Canny Edge Detection, to identify potential edges in the car images.
    - Use OpenCV's `findContours` function to extract contours from the processed images.
- **Shape Analysis:**
    - Analyze the extracted contours to identify and classify geometric features of the cars.
    - Calculate properties such as perimeter, area, and bounding rectangles to gather detailed shape information.
- **Visualization:**
    - Overlay the detected contours on the original images for visual representation.
    - Display the results using OpenCV's `imshow` function or save the processed images for documentation.
  
<br>
<br>


## IV. Conclusion
- This project demonstrated the application of contour detection techniques in analyzing car images.
- Key findings include:
    - Effective preprocessing significantly enhances contour detection accuracy.
    - Challenges such as noisy backgrounds and inconsistent lighting were addressed through robust image preprocessing techniques.
    - The extracted contours provide valuable insights into the structural features of vehicles, supporting applications like automated vehicle recognition.
- Challenges faced:
    - Variability in image quality and lighting conditions.
    - Fine-tuning the edge detection thresholds for optimal results.
- Outcomes:
    - Successfully extracted and analyzed contours from car images.
    - Highlighted the importance of preprocessing in achieving high-quality results.

<br>
<br>


## V. Additional Materials

### Part 1: 16 Basic OpenCV Projects
  ```python
  !git clone https://github.com/KanFudz/OpenCV_Finals_Mexe4102_JohnReiR.Malata_ArianeMaeD.Umali.git
  %cd OpenCV_Finals_Mexe4102_JohnReiR.Malata_ArianeMaeD.Umali
  from IPython.display import clear_output
  clear_output()
  ```

1. **Converting Images to Grayscale**
    - Use the color space conversion code to convert RGB images to grayscale for basic image preprocessing.
      ```python
      import cv2
      from google.colab.patches import cv2_imshow
      
      #colorful image - 3 channels
      image = cv2.imread("OPENCVPICS/cat.jpg")
      print(image.shape)
      
      #grayscale image
      gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
      print(gray.shape)
      cv2_imshow(gray)
      ```

2. **Visualizing Edge Detection**
    - Apply the edge detection code to detect and visualize edges in a collection of object images.


3. **Demonstrating Morphological Erosion**
    - Use the erosion code to show how an image's features shrink under different kernel sizes.


4. **Demonstrating Morphological Dilation**
    - Apply the dilation code to illustrate how small gaps in features are filled.


5. **Reducing Noise in Photos**
    - Use the denoising code to clean noisy images and compare the before-and-after effects.


6. **Drawing Geometric Shapes on Images**
- Apply the shape-drawing code to overlay circles, rectangles, and lines on sample photos.


7. **Adding Text to Images**
    - Use the text overlay code to label images with captions, annotations, or titles.


8. **Isolating Objects by Color**
    - Apply the HSV thresholding code to extract and display objects of specific colors from an image.


9. **Detecting Faces in Group Photos**
    - Use the face detection code to identify and highlight faces in group pictures.


10. **Outlining Shapes with Contours**
    - Apply the contour detection code to outline and highlight shapes in simple object images.


11. **Tracking a Ball in a Video**
    - Use the HSV-based object detection code to track a colored ball in a recorded video.


12. **Highlighting Detected Faces**
    - Apply the Haar cascade face detection code to identify and highlight multiple faces in family or crowd photos.


13. **Extracting Contours for Shape Analysis**
    - Use contour detection to analyze and outline geometric shapes in hand-drawn images.


14. **Applying Image Blurring Techniques**
    - Demonstrate various image blurring methods (Gaussian blur, median blur) to soften details in an image.


15. **Segmenting Images Based on Contours**
    - Use contour detection to separate different sections of an image, like dividing a painting into its distinct elements.


16. **Combining Erosion and Dilation for Feature Refinement**
    - Apply erosion followed by dilation on an image to refine and smooth out small features.

<br>

### Part 2: Revised Topic of Basic OpenCV
- Topic: Extracting Contours for Shape Analysis
    - Use contour detection to analyze and outline geometric shapes in hand-drawn images.
- Revised Topic: Extracting Contours for Shape Analysis in Car Images
    - Use contour detection to analyze and outline geometric shapes in car images.

<br>
<br>



## VI. References
- https://www.kaggle.com/datasets/lachin007/drawaperson-handdrawn-sketches-by-children
<br>
<br>


## VII. Group Members
<div align="center">

<table>
  <tr>
    <td><img src="https://github.com/user-attachments/assets/2d9ebaa0-d550-4b60-856d-d2c98fb9f3d1" alt="Malata" style="height: 230px; float: left;"></td>
    <td><img src="https://github.com/yannaaa23/CSE-Testing-Feb-19/blob/6ef6454fddba5503da2057bcf06fe77ca1491e0c/IMG_20230605_215028_860.jpg" alt="Umali" style="height: 230px; float: left;"></td>
  </tr>
  <tr>
    <td align="center"><strong>Malata, John Rei R.</strong></td>
    <td align="center"><strong>Umali, Ariane Mae D.</strong></td>
  </tr>
</table>

</div>

<br>
<br>



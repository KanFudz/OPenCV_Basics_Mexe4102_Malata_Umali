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
     <img src="https://drive.google.com/uc?export=view&id=116FACPAmeGTJJnXDbm5bFCNshqMdajiA" width="600"/>

2. **Visualizing Edge Detection**
  - Apply the edge detection code to detect and visualize edges in a collection of object images.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    image = cv2.imread("OPENCVPICS/obj.jpg")
    # cv2_imshow(image)
    
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

    canny_image = cv2.Canny(gray,150, 200)
    cv2_imshow(canny_image)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1zo0L2KNpboMpizSB5Y_g-p2_0R7dNqle" width="600"/>
    

3. **Demonstrating Morphological Erosion**
  - Use the erosion code to show how an image's features shrink under different kernel sizes.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    image = cv2.imread("OPENCVPICS/cat.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray,150, 200)
    kernel = np.ones((1,2), np.uint8)
    
    #Erosion
    erode_image = cv2.erode(canny_image,kernel, iterations=1)
    cv2_imshow(erode_image)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1V5eymGkKQB7R08nc8qsT3mDZpdFNghjm" width="600"/>


4. **Demonstrating Morphological Dilation**
  - Apply the dilation code to illustrate how small gaps in features are filled.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    image = cv2.imread("OPENCVPICS/cat.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray,150, 200)
    kernel = np.ones((5,5), np.uint8)
    #Dilation
    dilate_image = cv2.dilate(canny_image, kernel, iterations=1)
    cv2_imshow(dilate_image)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1ERGpKZUmTQATy4rFXntZRpiKXPWt7i2G" width="600"/>


5. **Reducing Noise in Photos**
  - Use the denoising code to clean noisy images and compare the before-and-after effects.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    image = cv2.imread("OPENCVPICS/us.jpg")
    # cv2_imshow(image)
    dst = cv2.fastNlMeansDenoisingColored(image, None, 50, 20, 7, 15)
    
    display = np.hstack((image, dst))
    cv2_imshow(display)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1yQ9MyTZs4m5FMD5etStf1BpyPCiwKFAJ" width="800"/>


6. **Drawing Geometric Shapes on Images**
  - Apply the shape-drawing code to overlay circles, rectangles, and lines on sample photos.
    ```python
    import cv2
    import numpy as np
    from google.colab.patches import cv2_imshow
    
    img = np.zeros((512, 512, 3), np.uint8)
    #uint8: 0 to 255
    
    # Drawing Function
    # Draw a Circle
    cv2.circle(img, (100,100), 50, (0,255,0),5)
    # Draw a Rectangle
    cv2.rectangle(img,(200,200),(400,500),(0,0,255),5)
    # Displaying the Image
    cv2_imshow(img)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1njOHqJXP8m8ecJWg51tA8Zb1XYGRbn7x" width="600"/>


7. **Adding Text to Images**
  - Use the text overlay code to label images with captions, annotations, or titles.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    
    # Load the photo
    image = cv2.imread("OPENCVPICS/dbed.jpg")
    
    # Check if the image was loaded
    if image is None:
        print("Error: Image not found. Check the file path.")
    else:
        print("Image loaded successfully!")
    
        # Add text to the image
        cv2.putText(
            image,
            "Smoke Then F...inals",        # Text to overlay
            (50, 50),                     # Position (x, y)
            cv2.FONT_HERSHEY_COMPLEX,     # Font style
            1,                            # Font scale
            (0, 255, 255),                # Text color (BGR: Yellow)
            2,                            # Thickness
            cv2.LINE_AA                   # Anti-aliasing for smooth text
        )
    
        # Display the image with the text
        cv2_imshow(image)
    ```
    <img src="https://github.com/user-attachments/assets/f63f314e-cf7b-417d-9fbc-12d249f3c90b" width="600"/>


8. **Isolating Objects by Color**
  - Apply the HSV thresholding code to extract and display objects of specific colors from an image.
    ```python
    import cv2
    import numpy as np
    from google.colab.patches import cv2_imshow
    #BGR Image . It is represented in Blue, Green and Red Channels...
    image = cv2.imread("OPENCVPICS/810.png")
    hsv = cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    
    # Blue Color Triangle
    lower_hue = np.array([65,0,0])
    upper_hue = np.array([110, 255,255])
    
    # Red Color
    lower_hue = np.array([0,0,0])
    upper_hue = np.array([20,255, 255])
    
    # Green Color
    lower_hue = np.array([46,0,0])
    upper_hue = np.array([91,255,255])
    
    # Yellow Color
    lower_hue = np.array([21,0,0])
    upper_hue = np.array([45,255,255])
    
    mask = cv2.inRange(hsv,lower_hue,upper_hue)
    cv2_imshow(mask)
    result = cv2.bitwise_and(image, image, mask = mask)
    cv2_imshow(result)
    cv2_imshow(image)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1t7MGAMaNnhJUkt91Qiy29TkGoLsSJ7iX" width="600"/>
    <img src="https://drive.google.com/uc?export=view&id=1p6XnL8hnXy30R6wQROtppT-7YU2b4nLU" width="600"/>
    <img src="https://drive.google.com/uc?export=view&id=1QZPNzCVpuDUZJyXm-l4rwbdy7KOqlRQg" width="600"/>



9. **Detecting Faces in Group Photos**
  - Use the face detection code to identify and highlight faces in group pictures.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    # img = cv2.imread("images/person.jpg")
    img = cv2.imread("OPENCVFACES/groupie1.jpg")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    faces = face_cascade.detectMultiScale(gray,1.3,5)
    # print(faces)
    for (x,y,w,h) in faces:
      cv2.rectangle(img,(x,y),(x+w,y+h),(0,255,0),3)
    
    cv2_imshow(img)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1uByKRzC5RmBrxe55y6onsMq0adQEy3IK" width="800"/>


10. **Outlining Shapes with Contours**
  - Apply the contour detection code to outline and highlight shapes in simple object images.
    ```python
    import cv2
    import numpy as np
    from google.colab.patches import cv2_imshow
    
    img = cv2.imread("OPENCVPICS/810.png")
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    
    ret, thresh = cv2.threshold(gray,50,255,1)
    contours,h = cv2.findContours(thresh,1,2)
    # cv2_imshow(thresh)
    for cnt in contours:
      approx = cv2.approxPolyDP(cnt,0.01*cv2.arcLength(cnt,True),True)
      n = len(approx)
      if n==6:
        # this is a hexagon
        print("We have a hexagon here")
        cv2.drawContours(img,[cnt],0,255,10)
      elif n==3:
        # this is a triangle
        print("We found a triangle")
        cv2.drawContours(img,[cnt],0,(0,255,0),3)
      elif n>9:
        # this is a circle
        print("We found a circle")
        cv2.drawContours(img,[cnt],0,(0,255,255),3)
      elif n==4:
        # this is a Square
        print("We found a square")
        cv2.drawContours(img,[cnt],0,(255,255,0),3)
    cv2_imshow(img)
    ```
    We found a square <br>
    We have a hexagon here <br>
    We found a triangle <br>
    We found a circle <br>
    We found a square <br>

    <img src="https://drive.google.com/uc?export=view&id=11tuzDc8-Ia-wA0tN_8r1a63WcM2q2K48" width="600"/>


11. **Tracking a Ball in a Video**
  - Use the HSV-based object detection code to track a colored ball in a recorded video.
    ```python
    import cv2
    import numpy as np
    from google.colab.patches import cv2_imshow
    import time  # For adding delays between frames
    
    # Initialize the video and variables
    ball = []
    cap = cv2.VideoCapture("OPENCVVID/vid.mp4")
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
    
        # Convert frame to HSV and create a mask for the ball color
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_hue = np.array([21, 0, 0])  # Adjust for your ball's color
        upper_hue = np.array([45, 255, 255])
        mask = cv2.inRange(hsv, lower_hue, upper_hue)
    
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        center = None
    
        if len(contours) > 0:
            # Get the largest contour
            c = max(contours, key=cv2.contourArea)
            ((x, y), radius) = cv2.minEnclosingCircle(c)
            M = cv2.moments(c)
    
            try:
                # Calculate the center of the ball
                center = (int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"]))
                # Draw a circle at the center
                cv2.circle(frame, center, 10, (255, 0, 0), -1)
                ball.append(center)
            except ZeroDivisionError:
                pass
    
            # Draw the tracking path
            if len(ball) > 2:
                for i in range(1, len(ball)):
                    cv2.line(frame, ball[i - 1], ball[i], (0, 0, 255), 5)
    
        # Display the frame in the notebook
        cv2_imshow(frame)
    
        # Add a small delay to simulate real-time playback
        time.sleep(0.05)
    
    cap.release()
    ```
    <img src="https://drive.google.com/uc?export=view&id=109gIsYdLwHG1SrYnG9SQmg9gcHyeqfNP" width="600"/>
    <img src="https://drive.google.com/uc?export=view&id=1Pr7hXx50CUMzkrbDo5Y5z5GK1dFe9omM" width="600"/>



12. **Highlighting Detected Faces**
  - Apply the Haar cascade face detection code to identify and highlight multiple faces in family or crowd photos.

    ```python
    !pip install face_recognition
    ```


    ```python
    import face_recognition
    import numpy as np
    from google.colab.patches import cv2_imshow
    import cv2
    
    # Creating the encoding profiles
    face_1 = face_recognition.load_image_file("OPENCVFACES/angel1.jpg")
    face_1_encoding = face_recognition.face_encodings(face_1)[0]
    
    face_2 = face_recognition.load_image_file("OPENCVFACES/ari1.jpg")
    face_2_encoding = face_recognition.face_encodings(face_2)[0]
    
    face_3 = face_recognition.load_image_file("OPENCVFACES/jr1.jpg")
    face_3_encoding = face_recognition.face_encodings(face_3)[0]
    
    face_4 = face_recognition.load_image_file("OPENCVFACES/step1.jpg")
    face_4_encoding = face_recognition.face_encodings(face_4)[0]
    
    known_face_encodings = [
                            face_1_encoding,
                            face_2_encoding,
                            face_3_encoding,
                            face_4_encoding
    ]
    
    known_face_names = [
                        "Angel",
                        "Ariane",
                        "John Rei",
                        "Stephen"
    ```
    

    ```python
    file_name = "OPENCVFACES/maris1.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1oGKCRsq3dD6Gr5b5dhHCg9llgAfsCY_f" width="600"/>


    ```python
    file_name = "OPENCVFACES/jr2.png"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1RfxAUgVmD5zqMllHzNOyxl2bgrvhiGSa" width="600"/>


    ```python
    file_name = "OPENCVFACES/angel2.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1OthLbfGitUFDdZWcgAEsxq24U098ZI2s" width="600"/>

    ```python
    file_name = "OPENCVFACES/jr3.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://github.com/user-attachments/assets/274819bc-dc36-466b-994b-f6297aca14aa"/>


    ```python
    file_name = "OPENCVFACES/angel3.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1yZNhmYrgQopL72qcsCITX2ICv7jEIHTD" width="600"/>


    ```python
    file_name = "OPENCVFACES/ari2.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1dhcl1des946Ioonk7KLa-qkjL901ys0n" width="600"/>


     ```python
    file_name = "OPENCVFACES/ari3.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1ztv_4nR_JGbaOo6BXY5YnzKqI68ZDLyB" width="600"/>


     ```python
    file_name = "OPENCVFACES/anthony1.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://drive.google.com/uc?export=view&id=18QW90xmALgmB-P1NIFSMHL2KcsaTJBoY" width="600"/>


     ```python
    file_name = "OPENCVFACES/step2.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1Q4QYQ8qU9P7DizCfdwnnPNFkiTixPYIg" width="600"/>


    ```python
    file_name = "OPENCVFACES/step3.jpg"
    unknown_image = face_recognition.load_image_file(file_name)
    unknown_image_to_draw = cv2.imread(file_name)
    
    face_locations = face_recognition.face_locations(unknown_image)
    face_encodings = face_recognition.face_encodings(unknown_image, face_locations)
    
    for (top,right, bottom, left), face_encoding in zip(face_locations, face_encodings):
      matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
    
      name = "Viral"
    
      face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
      best_match_index = np.argmin(face_distances)
      if matches[best_match_index]:
        name = known_face_names[best_match_index]
      cv2.rectangle(unknown_image_to_draw, (left, top), (right, bottom),(0,255,0),3)
      cv2.putText(unknown_image_to_draw,name, (left, top-20), cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),2, cv2.LINE_AA)
    
    cv2_imshow(unknown_image_to_draw)
    ```
    <img src="https://github.com/user-attachments/assets/9b600ebb-1e2b-44ee-9272-26a36e00008f"/>
    
13. **Extracting Contours for Shape Analysis**
  - Use contour detection to analyze and outline geometric shapes in car images.
    
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    # Read the input image
    image = cv2.imread("OPENCVPICS/HANDDRAWN.png")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw contours
    contour_image = image.copy()
    
    # Draw the contours on the image
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Analyze each contour and approximate the shape
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
    
        # Find the bounding rectangle to label the shape
        x, y, w, h = cv2.boundingRect(approx)
    
        # Determine the shape based on the number of vertices
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # Check if the shape is square or rectangle
            aspect_ratio = float(w) / h
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif len(approx) > 4:
            shape = "Circle"
        else:
            shape = "Polygon"
    
        # Put the name of the shape on the image
        cv2.putText(contour_image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Stack the original, edge-detected, and contour images for display
    stacked_result = np.hstack((cv2.resize(image, (300, 300)),
                                cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (300, 300)),
                                cv2.resize(contour_image, (300, 300))))
    
    # Display the results
    cv2_imshow(stacked_result)
    ```
    <img src="https://github.com/user-attachments/assets/4263b74f-b886-4242-8a40-d25d33a325a4" width="800"/>


14. **Applying Image Blurring Techniques**
  - Demonstrate various image blurring methods (Gaussian blur, median blur) to soften details in an image.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    image = cv2.imread("OPENCVPICS/jr.jpg")
    Gaussian = cv2.GaussianBlur(image,(7,7),0)
    Median = cv2.medianBlur(image,5)
    
    display = np.hstack((Gaussian,Median))
    cv2_imshow(display)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1qRPx8LXWMavVIGXlkbgqNMfnCkY_0wr5" width="800"/>


15. **Segmenting Images Based on Contours**
  - Use contour detection to separate different sections of an image, like dividing a painting into its distinct elements.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    # Read the input image
    image = cv2.imread("OPENCVPICS/contour.jpg")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply binary thresholding
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a blank mask for segmentation
    segmented_image = np.zeros_like(image)
    
    # Loop through each contour to extract and display segmented areas
    for i, contour in enumerate(contours):
        # Create a mask for the current contour
        mask = np.zeros_like(gray)
        cv2.drawContours(mask, [contour], -1, 255, -1)
    
        # Extract the segment by masking the original image
        segmented_part = cv2.bitwise_and(image, image, mask=mask)
    
        # Add the segment to the segmented image
        segmented_image = cv2.add(segmented_image, segmented_part)
    
        # Optionally draw bounding boxes for visualization
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
    
    
    # Display results
    cv2_imshow(image)  # Original image with bounding boxes
    cv2_imshow(segmented_image)  # Segmented image
    ```
    <img src="https://drive.google.com/uc?export=view&id=1LsMLpS0xF21Anb1w3X_znJcw5_50ph3V" width="800"/>


16. **Combining Erosion and Dilation for Feature Refinement**
  - Apply erosion followed by dilation on an image to refine and smooth out small features.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    image = cv2.imread("OPENCVPICS/as.jpg")
    gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    canny_image = cv2.Canny(gray,150, 200)
    kernel = np.ones((1,1), np.uint8)
    erode_image = cv2.erode(canny_image,kernel, iterations=1)
    kernel1 = np.ones((3,3), np.uint8)
    dilate_image = cv2.dilate(erode_image, kernel1, iterations=1)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(canny_image, 'Canny Image', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(erode_image, 'Eroded', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(dilate_image, 'Feature Refined', (10, 30), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
    
    display = np.hstack((canny_image,erode_image,dilate_image))
    cv2_imshow(display)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1rNlKAYkNzsi_Oy-qbq9xempAnRtAU22l" width="800"/>

<br>


### Part 2: Revised Topic of Basic OpenCV
- Topic: Extracting Contours for Shape Analysis
    - Use contour detection to analyze and outline geometric shapes in hand-drawn images.
- Revised Topic: Extracting Contours for Shape Analysis in Car Images
    - Use contour detection to analyze and outline geometric shapes in car images.

    ```python
    !git clone https://github.com/KanFudz/OpenCV_Finals_Mexe4102_JohnReiR.Malata_ArianeMaeD.Umali.git
    %cd OpenCV_Finals_Mexe4102_JohnReiR.Malata_ArianeMaeD.Umali
    from IPython.display import clear_output
    clear_output()
    ```
13. **Extracting Contours for Shape Analysis**
    - Use contour detection to analyze and outline geometric shapes in hand-drawn images.
    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    # Read the input image
    image = cv2.imread("DATASET/audi/7.jpg")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw contours
    contour_image = image.copy()
    
    # Draw the contours on the image
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Analyze each contour and approximate the shape
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
    
        # Find the bounding rectangle to label the shape
        x, y, w, h = cv2.boundingRect(approx)
    
        # Determine the shape based on the number of vertices
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # Check if the shape is square or rectangle
            aspect_ratio = float(w) / h
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif len(approx) > 4:
            shape = "Circle"
        else:
            shape = "Polygon"
    
        # Put the name of the shape on the image
        cv2.putText(contour_image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Stack the original, edge-detected, and contour images for display
    stacked_result = np.hstack((cv2.resize(image, (300, 300)),
                                cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (300, 300)),
                                cv2.resize(contour_image, (300, 300))))
    
    # Display the results
    cv2_imshow(stacked_result)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1GB4n3PGCkdbTfDjss1_ZGmPIUBArIV_R" width="800"/>


    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    # Read the input image
    image = cv2.imread("DATASET/mercedes/7.jpg")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw contours
    contour_image = image.copy()
    
    # Draw the contours on the image
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Analyze each contour and approximate the shape
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
    
        # Find the bounding rectangle to label the shape
        x, y, w, h = cv2.boundingRect(approx)
    
        # Determine the shape based on the number of vertices
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # Check if the shape is square or rectangle
            aspect_ratio = float(w) / h
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif len(approx) > 4:
            shape = "Circle"
        else:
            shape = "Polygon"
    
        # Put the name of the shape on the image
        cv2.putText(contour_image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Stack the original, edge-detected, and contour images for display
    stacked_result = np.hstack((cv2.resize(image, (300, 300)),
                                cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (300, 300)),
                                cv2.resize(contour_image, (300, 300))))
    
    # Display the results
    cv2_imshow(stacked_result)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1zKQr_rDaMfqa9SibJ0HhgRLfn_Y11kVy" width="800"/>


    ```python
    import cv2
    from google.colab.patches import cv2_imshow
    import numpy as np
    
    # Read the input image
    image = cv2.imread("DATASET/lamborghini/7.jpg")
    
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply edge detection
    edges = cv2.Canny(gray, 50, 150)
    
    # Find contours
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Create a copy of the original image to draw contours
    contour_image = image.copy()
    
    # Draw the contours on the image
    cv2.drawContours(contour_image, contours, -1, (0, 255, 0), 2)
    
    # Analyze each contour and approximate the shape
    for contour in contours:
        # Approximate the contour
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
    
        # Find the bounding rectangle to label the shape
        x, y, w, h = cv2.boundingRect(approx)
    
        # Determine the shape based on the number of vertices
        if len(approx) == 3:
            shape = "Triangle"
        elif len(approx) == 4:
            # Check if the shape is square or rectangle
            aspect_ratio = float(w) / h
            shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
        elif len(approx) > 4:
            shape = "Circle"
        else:
            shape = "Polygon"
    
        # Put the name of the shape on the image
        cv2.putText(contour_image, shape, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Stack the original, edge-detected, and contour images for display
    stacked_result = np.hstack((cv2.resize(image, (300, 300)),
                                cv2.resize(cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR), (300, 300)),
                                cv2.resize(contour_image, (300, 300))))
    
    # Display the results
    cv2_imshow(stacked_result)
    ```
    <img src="https://drive.google.com/uc?export=view&id=1aiF8HKd8hq4IOSw23P87ZhdtGGF04YnX" width="800"/>



<br>
<br>



## VI. References
- https://www.kaggle.com/datasets/lachin007/drawaperson-handdrawn-sketches-by-children
- https://www.pexels.com/photo/brown-tabby-cat-2071882/
- https://myenglishgrammar.com/wp-content/uploads/2023/10/objects-group.png
- https://miro.medium.com/v2/resize:fit:1100/format:webp/0*nfqCa_X1oL66h-kG
- https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSiNtiSPQ9EGnKivtoQvvUpHOEuRxedjpAQFw&s
- https://www.facebook.com/photo.php?fbid=558271683638013&id=100083655572387&set=a.242669278531590
- https://images.app.goo.gl/7eni9PwEggrc5Jxf7
- https://www.erinhanson.com/content/inventoryimages/Erin-Hanson-Cypress-Mosaic-.jpg
- https://youtu.be/E3Lg4aZVCAU?si=_UT8JlruRT_q7wi1
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



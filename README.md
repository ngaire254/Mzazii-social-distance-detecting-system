# Mzazii-social-distance-detecting-system
A social distance detecting system using python and yolo
CHAPTER 1: INTRODUCTION
1.1 BACKGROUND OF STUDY
During the ongoing COVID-19 pandemic, many institutions have implemented numerous social distancing measures in travel, public places and warning individuals to keep 1.5-2 meters from each other. The goal of the proposed project is to detect when the social distancing rule is violated by one person being close to one another. By the use of a camera, the system will be able to detect the human beings and determine the distance between one person to another.
The reason behind the creation of the smart social distance detecting system, is to reduce the rate of COVID-19 transmission in the society. Social distancing has been known as one of the preventive measures that can be taken to reduce the spread of the virus [1]. COVID-19 can spread through coughing, sneezing and close contact. By minimizing the amount of close contact, we have with others, we reduce our chances of catching the virus and spreading it to our loved ones and within our community [2].
Smart social distance detecting system can be used in bus stop, railway terminals and streets where people are mostly congested. The system will reduce on the personnel that can be used in determining the distance from one individual to the other. The system also will be highly accurate compared to a human bare eye which cannot automatically detect the distance between one individual to the other. The amount of time taken by the smart system to detect less distance from one person to another is less compared to that of a human. The system also can easily detect the distance between moving individuals which can be hard to detect by a human.
Due to these discussed features, machine learning plays a central role in social distance monitoring. In this work, a deep learning framework is presented to maintain social distance between individuals in a public campus environment by utilizing a top view perspective. For human detection, deep learning architecture, i.e., YOLO. After the detection of human bounding box information, the central point, also called the centroid detected bounding box, is calculated. To measure the distance between people in the input image, the Euclidean distance is computed between each detected centroid. Using the distance to pixel approximation method, a distance threshold is defined for minimum social distance violation. The detected distance value is checked with a specified threshold value; either it appears under the violation threshold or not. The color of the bounding box is initialized as green; if its distance values appear in the violation list, its color is changed to red.
An audio-visual cue is emitted each time an individual breach of social distancing is detected. The system will also help to show the over congested regions in an institution.
Researchers have made different efforts [3], [4], [5], to develop efficient methods for social distance monitoring. They utilized different machine and deep learning-based approaches to monitor and measured the social distancing among people. Authors use various clustering and distance-based approaches to determine the distance between people; however, they mostly concentrated on the side or frontal view perspectives whereby camera calibration is required to map distance to pixels information for practical and measurable units.
Kylie [6], examined the correlation linking the social distancing strictness and the region's economic condition and reported that modest steps of this exercise could be adopted for avoiding a massive outbreak. Thus, until now, several nations have adopted solutions based on advanced technology to defeat the pandemic loss. Many advanced nations are exercising Global Positioning System technology to control suspected and infected people's movements. Nguyen [7] presented a review of many developing technologies, consisting of Bluetooth, Wi-fi, GPS, mobile phones, computer vision, image processing, and deep learning that performed a vital function in various possible social distancing situations. Few employed drones and different monitoring devices to identify people gatherings.

1.2 PROBLEM STATEMENT
Social distancing has been identified as the most important practice since late December 2019, after the growth of the COVID-19 pandemic. It is opted as standard practice to stop the infectious virus transmission. During the first week of February 2020, the number of cases was increasing on an exceptional basis, with reported cases ranging from 2000 to 4000 per day. Later, there was a relief symbol for the first time, with no new positive cases for five successive days until March 23, 2020. This happens because the social distance exercise, which was started in China and, lately, utilized globally to control the spread of COVID-19.
Taking advantage of deep learning approaches, researchers presented efficient solutions for social distance monitoring. Utilizing the YOLO with Deep sort algorithm to identify and track people; they practiced an Open Image Data set repository. The scholars also examined results with other deep learning paradigms. Therefore, inspired by this research, I came up with an idea of developing a system that allows a better view of the scene and overcomes occlusion concerns by performing a pivotal role in social distance monitoring.
1.3 OBJECTIVES
1.3.1 General Objective
The main goal for developing smart social distance detecting system is to ensure that people are able to adhere to the regulation set in order to reduce the rate of transmitting COVID-19. Person detection and social distance monitoring have to be put into consideration in order for the main purpose of the project to be attained.
1.3.2 Specific Objective
i.	To acquire the distance between two individuals by the use of pre-trained YOLO for human detection in bird view perspective and calculating information of bounding box centroid. 
ii.	To design a distance-based violation threshold for social distance utilizing a pixel to distance estimation approach to monitor people's interaction.
iii.	To develop a social distance monitoring system utilizing deep learning architecture. The developed system is used as a prevention measuring tool to reduce close interaction between peoples and limit COVID-19 virus spread.
iv.	To simulate the social distance detecting system by the use of a video stream with human beings.


1.4 SIGNIFICANCE OF THE STUDY
The main reason for proposing a smart social distance detecting system is to enable the people to be able to cope with the regulations set for preventing the spread of COVID-19. Maintaining a distance of 1.5-2 meters from one person to another, has been named to be one of the best ways to prevent the spread of the virus.
Using a smart system will be much efficient than a human being, the smart system will detect violation of the minimum set distance between two individuals faster and more accurate. It will be more effective as the smart system can detect social distance violation to a large group of people than a human who is limited to detecting two individuals at a time.

1.5 SCOPE OF THE STUDY
The project aims at detecting the project aims at detecting people who are violating the set COVID-19 protocols for social distancing in streets and in the organizations.
The proposed system uses YOLO to detect only humans and a calibrated camera to calculate the distance between individuals.

1.6 LIMITATIONS
Even though the system is able to detect people within its range, some errors may occur hindering the system from making the correct judgement. Some of the limitations are caused by the overlapping of frames or when people are walking too to each other. Overlapping of frames reduces the number of people who are detected by the system as some individuals are blocked by others from the camera.






CHAPTER 2: LITERATURE REVIEW
2.1 INTRODUCTION
Literature review is a survey of scholarly sources on a specific topic that provides an overview of current knowledge, allowing you to identify relevant theories, methods and gaps in existing research [8].
For the importance of security and distance measuring in several fields, different approaches have been developed. Machine learning and camera calibration has been used.

2.2 EXISTING SYSTEMS
2.2.1 Human Detection Using YOLO Algorithms
YOLO stands for ‘You Only Look Once’ and this algorithm is an excellent object-detection algorithm that applies convolutional neural networks (CNN).
CNNs are Artificial Neural Networks that use pixel data for image recognition. Advanced implementations of deep learning that are based on machine vision use CNNs for generative tasks. Similarly, the YOLO algorithm uses CNNs for real-time object detection. It is one of the fastest algorithms for this purpose, capable of working on visual data at up to 155 frames per second.
Yolo has made it possible to develop face recognition systems which may be used for surveillance and security purposes.
As a result, this empowers the systems to monitor the footage in real-time and can be a pathbreaking development in regards to public safety.
Manual monitoring of a CCTV camera requires constant human intervention so they are prone to errors and fatigue. AI-based surveillance is automated and works 24/7, providing real-time insights.
In 2018, Tossaporn Santad and his fellow researchers [9] came up with a system that applied YOLO to detect abandoned luggage by the use of deep learning neural networks. Then, the system can track movement of people and their baggage. They designed a graphical user interface for alarming and retrieving pass port events. The system would alarm whenever a bag had been abandoned in a given location. The system worked with varying lighting conditions and under different camera locations.
YOLO has also been used in the development of a system for detecting of moving objects in a metro rail cctv video [11]. The system focuses on the detection of moving objects from stationary surveillance cameras in metro stations and alerts when an object crosses the yellow line. For the detection of an object, YOLO object detection models are used. To make it simple, pre-trained models are used for object detection in this project. The processing steps of motion detection for video surveillance include detection of motion, classification and object detection, behavior understanding, and activity recognition. The system works in real time with more accuracy than previous models available. The color detection method is used for detecting the yellow lines in the metro stations and since the camera is static, line detection is performed only on the first frame which further reduces the time of detection of objects. An emergency alert is given to the authorities/ passengers in the form of sound signals once the object crosses the yellow line in the stations.

Advantages of using YOLO algorithms in human detection
•	YOLO algorithms have a high detection speed.
•	YOLO is a very accurate and has predictive technique whose output has minimal background errors and is barely bothered by background noise.
•	The YOLO algorithm has and incredible capability to learn the representation of objects.

2.2.2 Measuring the Distance Between Object in a Video Stream
Measuring the distance between objects in a video stream is one of the major aspects in the development of the social distance detecting system. Earlier several systems have been developed applying the same technique of measuring the distance between object either on a video stream or on an image. In the proposed system, the aspect of measuring distance will be used to measure the distance between individuals in the video steam.
In 2020, Jae moon lee developed [10] came up with a new method of measuring distance using a camera and artificial intelligence. Based on the fact that one can measure the distance between a camera and the object under focus, it then from where he bases his technique. In jae’s technique, photos of an object are taken from two separate locations, and object detection - a sort of Artificial Intelligence technology - is used to measure the distance between it and the camera.
Several equations were induced mathematically to ensure the accuracy of the proposed method. The iPhone 8 plus was used to ensure that the process was real-time. Google TensorFlow-lite object detection model was used. Several experiments in various environments were conducted. Empirical data had illustrated that such distance measurements using object detection modules are not yet precise and accurate. One of the main reasons for this is that object detection technology cannot yet measure accurately the size of detected objects. This is an issue to be resolved relatively easily with the advancement of Artificial Intelligence in the near future. The proposed method and developed system can be used to aid the visually impaired. If the system is incorporated in guidance systems for the visually impaired, it could effectively alert its user of obstacles that are in the way. The system may not provide a precise and accurate measurement of distance, but the information it provides will be more than enough for the user to avoid colliding into obstacles.
2.2.3 Proposed System
In this project, pre-trained YOLO will be tested for measuring distance between individuals. The data used for testing will be a video stream from which the system will detect human beings and measure the distance between them. The measured distance will then be used to determine whether the social distance rule has been violated or not.

2.3 EXISTING DEVELOPMENT TOOLS
2.3.1: Python Programming Language
Python is an interpreted, object-oriented, high level programming language with dynamic semantics. Its high-level built-in data structures, combined with dynamic typing and dynamic binding makes it very attractive for Rapid application Development, as well as for use as a scripting or glue language to connect existing components together [12].
2.3.2 Machine Learning
Machine learning is an evolving branch of computational algorithms that are designed to emulate human intelligence by learning from the surrounding environment. Techniques based on machine learning have been applied successfully in diverse fields ranging from pattern recognition, computer vision, spacecraft engineering, finance, entertainment, and computational biology to biomedical and medical applications [13].

2.4 JUSTIFICATION
Literature review summarizes and synthesizes the arguments and ideas of existing human detecting systems and measurement of distance using artificial intelligence. With profound knowledge of the gaps exposed in the existing systems proposed system will overpower them.
Python programming will be used to develop the social distance because its selection of machine learning-specific libraries and frameworks simplify development process and cut development time. Python has a simple syntax and its readability promote rapid testing of algorithms.

2.5.  CONCLUSION
According to the given literature review, many human detection systems have been build using different detection algorithms, each with its specific advantages and disadvantages in comparison with other techniques. However, none of the accomplished studies described the applications of human detection system in measuring distance between individuals.
With traditional methods not being effective and time consuming, use of machine learning approaches proves to be an important aspect for shaping the automation of social distance detection system. Detection of violation of social distance involves different things, it involves human detection and measuring of distance between the two detected individuals.
CHAPTER 3: RESEARCH METHODOLOGY
3.1 INTRODUCTION
Research methodology is a way to systematically solve a research problem following specific procedures and techniques. Methodology allows one to critically evaluate study’s overall validity and reliability [14].
This topic discusses the methods that will be used to collect data and how the collected data will be analyzed. Data will be collected from both primary and secondary sources.
3.2 DATA COLLECTION TECHNIQUES
3.2.1 Questionnaires
Questionnaires are major instruments of collecting data in survey research. They are set of standardized questions, often called items, that follow a fixed scheme in order to collect individuals’ data about a given topic [15].  Both open-ended and closed questions will be used.

Advantages
•	Result into wide range of views from customers.
•	Questionnaires are the most affordable ways to gather quantitative data.
•	It’s easy and quick to collect results.
•	When data has been quantified it can be used to compare and contrast other research and maybe used to measure change.


Disadvantages
•	There is a chance that some questions will be ignored and left unanswered.
•	Differences in understanding and interpretation.
•	Questionnaire cannot fully capture emotional responses and feelings.

3.2.2 Interview
Interview is a qualitative research technique which involves asking open-ended questions to converse with respondents and collect elicit data about a subject [16].Type of interviews include; Personal interview where questions are asked personally directly to the respondent it gives a higher response rate. Telephonic interviews are widely used and easy to combine with online surveys to carry out research effectively. Email or web-page interview; since online research is growing and more consumers are migrating to more virtual world e-mail and web-page interviews are efficient.

Advantages of using interviews
•	I was able to gain valuable insights based on the depth of the information gathered and the wisdom.
•	Interviews require only simple equipment and build on conversation skills which researchers already have.
•	Interviews are more flexible.
•	Direct contact at the point of interview means data can be checked for accuracy and relevance are they are collected.

Disadvantages of Interviews
•	Data analysis and preparation can be difficult and time consuming.
•	Consistency and objectivity are hard to achieve.
•	Identity of researcher may affect the statements of the interviewee.
•	Some people may not show up for the interview.

3.2.3 Observations
It’s a technique that involves systematically selecting, watching, listening, reading, touching and recording behavior and characteristics of living beings, objects or phenomena [17].

Advantages
•	Data can be collected at the time they occur.
•	Observation study describe observed phenomena as they occur in natural setting.
•	Offers an opportunity for longitudinal analysis.
Disadvantages
•	Difficulties in quantification.
•	Sample size observed is usually small.
•	There is no opportunity to study the past when using observation method.

3.2.4 Justification
Since data collection is essential in research, to gather information in the proposed system observation and interviewing methods will be used.
Observations will be made on specific point in organization entrances and along the known crowded streets. It will enable the researcher to identify the times when the given areas are highly crowded.
Interviewing specific persons in organizations will enable one obtain information such as the factors that contribute to people violating the rules set for social distancing.
Through observation one is exposed to first-hand information and also helps in gaining more insights into current systems.
3.3 Software Development Technique
3.3.1 Rapid Application Development Methodology
RAD is an agile software development approach that focuses more on ongoing software projects and user feedback and less on following a strict plan [18].
RAD develops software via the use of prototypes, dummy, backend databases and its goal are to meet the business need of the system and customer is heavily involved in the process [19].
It consists of four phases [20]. 
Requirement analysis- Developers, clients and team members communicate to determine the goals and expectations for the project
User Design- involves building out user design through various prototype iterations
Rapid construction- Takes the prototypes and beta systems from design phase and converts them into a working model.
Cutover – implementation phase where finished product is launched.

 
Advantages of using RAD Methodology
•	RAD lets you break the project into smaller and more manageable tasks.
•	Task oriented structure allows project managers to optimize their team’s efficiency by assigning tasks according to members specialist and experience.
•	Clients get a working product delivered in a shorter time frame.
•	Regular communication and constant feedback between team members and stakeholders increases the efficiency of design and build process.

Disadvantages of RAD
•	Needs strong team collaboration.
•	Needs highly skilled developers.
•	Only suitable for projects which have a small development time.
•	Only systems which can be modularized can be developed using RAD.

3.3.2 Agile Methodology
Agile methodology is a type of project management process, mainly used for software development, where demands and solutions evolve through the collaborative effort of self-organizing and cross-functional teams and their customers [21].
It is used to deliver complex projects due to its adaptiveness. It emphasizes on collaboration, flexibility, continuous improvement and high-quality results.
The five phases are;
Project initiation which is about discussing project vision and ROI justification. Team members, time and work resources required are determined.
Planning- it is where the team gets together with their sponsor or product owner and identifies exactly what they are looking for.
Development –once requirements have been defined actual work begins.
Production –a handover with relevant training should take place between the production and support teams.
Retirement – it is the final stage. Customers are notified and informed about migration to newer releases or alternative options.
 
It has several frameworks such as;
Scrum used to implement the ideas behind agile software development
Kanban is a visual method used to paint picture of the workflow process, with an aim to identify any bottlenecks early in the process
FDD- Is a lightweight iterative and incremental software development process with an objective to deliver tangible, working software in timely manner.
Advantages of agile methodology [22]
•	Better product quality- agile methods have excellent safeguards to make sure that quality is as high as possible
•	Higher customer satisfaction- by keeping customers involved and engaged.
•	High team morale-being part of self-managing team allows people to be creative, innovative and acknowledged for their expertise.
•	Increased collaboration and ownership- development team, product owner and scrum master work closely together on a daily basis.

3.3.3 Justification
In the proposed project I will use agile software development methodology because; agile approach advocates building prototypes, testing, and incorporating feedback as soon as possible. 

3.4: SYSTEM REQUIREMENTS
3.4.1: Software Requirements
•	OPERATING SYSTEM: Windows 10 and higher version, Linux or MacOS
•	PROGRAMMING LANGUAGE: Python
3.4.2: Hardware Requirements
•	PROCESSOR: Intel Core I 3 and above
•	RAM: minimum of 4gb
•	Storage 128gb and above
•	Printer
3.4.3: Functional Requirements
Are function or features that must include in any system to be of help in preventing the spread of covid-19. The developed system has the following functional requirements;
•	The system is able to detect human beings.
•	The system can measure the distance between pairs of individuals.
•	Be able to detect whether individuals are violating to social distancing protocols.
3.4.4: Non-Functional Requirements
It’s a description of features, characteristics and attributes of the system as well as any constraints that may limit the boundaries of the proposed system. They are based on performance, information, control and security efficiency and services. Based on the developed system, non-functional requirements include;
•	The system provides better accuracy.
•	The system has a simple interface for users to use.
•	Perform efficiently in short amount of time.

3.5: CONCLUSION
Social distancing plays a critical role in controlling the spread of covid-19. With the help of the Smart social distancing detecting system, it will be easier for the government and the organizations to implement the covid-19 protocols. This will be a more effective and less time-consuming technique of implementing social distancing.






CHAPTER 4: SYSTEM DESIGN AND IMPLIMENTATION
4.1: INTRODUCTION
System design is the process of defining the architecture, product design, modules, interfaces, and data for a system to satisfy specified requirements [23].
4.2: SYSTEM DESIGN
The proposed system is intended to alert the user of people who are violating the social distancing rule based on the calculated distance between the individuals.
To develop the system, the following logical diagrams will be followed for defining the architecture.
4.2.1: Flowchart Diagram
When the system starts, the program loads the camera and reads the frames of the camera. It the checks whether there are any humans detected by the camera, if there are no humans detected it resumes to check. When a human being is detected, the system continues to look for a second human and measures the distance between them. If the distance between them is less than one meter, the system gives a sound alert, if the distance is more than one meter the system goes back to the first step of reading the frames of the camera.






	No




	Yes



	No

	Yes




4.2.2: UML Diagram
 
Step 1
In the first step, the system filters all other object that are captured in the video stream and only detects class people.
Step 2
The system computes the detected individuals in the system pairwise between their centroids.
Step 3
The system checks the distance between individuals and it is able to know whether the social distance rule has been violated by comparing the distance measured with programmed distance.



4.2.3: Use case diagram

 
The System acquires data by capturing a video stream of the scene, the objects on video stream are read, resized and filtered for better understanding by the system. The system removes other objects from detection and detects only class people. It then measures the distance between individuals and compares it with the programmed distance to detect violation.

4.2.3: Process Design
For proper working of the social distance detecting system the following system process must be involved;
The first process is video data acquisition, the video of a scene where the camera has been focused towards is captured. 
In the second process, the system reads the video frames used to capture the scene by the camera. In the third process, the system detects the class people capture in the video stream. It then goes to the fourth process of checking the number of people in the video stream. 
In the fifth process it determines the distance between the centroids of the bounding boxes. In the last process, it makes the decision whether the distance is safe or not according to the programmed distance.
 

4.3: IMPLIMENTATION APPROACHES
GPU-based systems
In recent years, manufacturers have been forced to find alternatives to the traditional source of computational power increase. Due to the fundamental limitation in the fabrication of integrated circuits, it is no longer feasible to rely on upward processor clock speeds as a means of extracting additional power from existing architecture. The release of Graphic Processor Units (GPUs) that possessed pipeline attracted many researchers to the possibility of using graphics hardware for many applications. With this in mind, NVIDIA released GPUs for professional applications in the market. NVIDIA expanded in many applications including the acceleration of AI and deep learning architectures. In addition to that, NVDIA provides an application programming interface, which is known as CUDA or Compute unified device architecture. It allows the creation of parallel computation, which utilizes GPUs. NVDIA released various developer kits such as Jetson nano, Jetson TX1, Jetson TX2, and Jetson AGX Xavier for edge computing. These devices are incredible for AI performance, which are targeted for real-time applications. In our research, we will utilize Jetson nano and Jetson AGX Xavier for execution of the proposed technique.
Social distancing on embedded systems
To assess the performance for the proposed techniques, specific tests have been carried out to examine the dependency of the proposed technique computational cost on the targeted hardware. We deployed the algorithm as standalone applications in two different NVIDIA devices: Jetson nano and Jetson AGX Xavier. V2 raspberry PI camera has been used and exposed it to another computer device, which displayed set of videos, which were obtained from thermal cameras. We performed the test measuring the maximum frames per seconds fps that can be obtained and we compared the achieved results with other methodologies. Results are reported in Table 3. Based on the results from our experiments, NVIDIA Jetson Xavier achieved the best results for frames per seconds, which is reaching 23 fps while the real time for NVIDIA Jetson nano has reached only 11 fps. In such configuration, the Xavier board is 2× times faster than the NVIDIA Jetson nano.
4.4 CODE DETAILS AND CODE EFFICIENCY
Importing necessary packages and reading the dataset
import cv2
import time
import imutils
import numpy as np
from numpy import sqrt 
import winsound
cv2: is a library of programming functions mainly aimed at real-time computer vision. Originally developed by Intel, it was later supported by Willow Garage then Itseez (which was later acquired by intel. The library is cross-platform and free for use under the open-source Apache 2 License. Starting with 2011, OpenCV features GPU acceleration for real-time operations.
Time: Python has a module named time to handle time-related tasks.
Imutils: A series of convenience functions to make basic image processing functions such as translation, rotation, resizing, skeletonization, displaying Matplotlib images, sorting contours, detecting edges, and much easier with OpenCV and both Python 2.7 and Python 3.
numpy: It is a library for the Python programming language, adding support for large, multi-dimensional arrays and matrices, along with a large collection of high-level mathematical functions to operate on these arrays.
Winsound: The winsound module provides access to the basic sound-playing machinery provided by Windows platforms.

Loading Caffe Model
#Loading Caffe Model
print('[Status] Loading Model...')
nn = cv2.dnn.readNetFromCaffe(MODEL_CONFIG, MODEL_WEIGHTS)
Initialize Video Stream
#Initialize Video Stream
print('[Status] Starting Video Stream...')
vs = VideoStream(src = 1).start()
time.sleep(0.1)
fps = FPS().start()
Resize Frame to 600 pixels
#Resize Frame to 600 pixels
    frame = vs.read()
    frame = imutils.resize(frame, width=600)
    frame = cv2.resize(frame, (0, 0), None, 1.5, 1.5)
Converting Frame to Blob
(H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
Passing Blob through network to detect and predict
#Passing Blob through network to detect and predict
    nn.setInput(blob)
    detections = nn.forward()
Creating dictionaries to store position and coordinates
#Creating dictionaries to store position and coordinates
    pos = {}
    coordinates = {}
Looping over the detections
#Loop over the detections
    for i in np.arange(0, detections.shape[2]):
Extracting the confidence of predictions
#Extracting the confidence of predictions
        conf = detections[0, 0, i, 2]
Extracting the index of the labels from the detection
#Extracting the index of the labels from the detection
            object_id = int(detections[0, 0, i, 1])

Storing bounding box dimensions
#Storing bounding box dimensions
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype('int')
Drawing the prediction on the frame
#Draw the prediction on the frame
                label = 'Person: {:.1f}%'.format(conf * 100)
                cv2.rectangle(frame, (startX, startY), (endX, endY), (10,255,0), 2)

                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, label, (startX, y), cv2.FONT_HERSHEY_DUPLEX, 0.5, (20,255,0), 1)
Adding the bounding box coordinates to dictionary
#Adding the bounding box coordinates to dictionary
                coordinates[i] = (startX, startY, endX, endY)
Extracting Mid point of bounding box
#Extracting Mid point of bounding box
                midX = abs((startX + endX) / 2)
                midY = abs((startY + endY) / 2)
Calculating height of bounding box
#Calculating height of bounding box
                ht = abs(endY-startY)
Calculating distance from camera
#Calculating distance from camera
                distance = (FOCAL_LENGTH * 165) / ht
Mid-point of bounding boxes in cm
#Mid-point of bounding boxes in cm
                midX_cm = (midX * distance) / FOCAL_LENGTH
                midY_cm = (midY * distance) / FOCAL_LENGTH
Appending the mid points of bounding box and distance between detected object and camera
#Appending the mid points of bounding box and distance between detected object and camera 
                pos[i] = (midX_cm, midY_cm, distance)
Looping over positions of bounding boxes in frame
#Looping over positions of bounding boxes in frame
    for i in pos.keys():
        for j in pos.keys():
            if i < j:
Calculating distance between both detected objects
#Calculating distance between both detected objects
                squaredDist = (pos[i][0] - pos[j][0])**2 + (pos[i][1] - pos[j][1])**2 + (pos[i][2] - pos[j][2])**2
                dist = sqrt(squaredDist)
Checking threshold distance - 175 cm and adding warning label
if dist < 175:
                    proximity.append(i)
                    proximity.append(j)
                    cv2.putText(frame, mixer.init(), (50,50), cv2.FONT_HERSHEY_DUPLEX, 0.5, [0,0,255], 1)
                    print(beep.play())
                    
    for i in pos.keys():
        if i in proximity:
            color = [0,0,255]
        else:
            color = [0,255,0]

Drawing rectangle for detected objects
#Drawing rectangle for detected objects
        (x, y, w, h) = coordinates[i]
        cv2.rectangle(frame, (x, y), (w, h), color, 2)                      
    cv2.imshow('Live', frame)     
    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):             
        break
    fps.update()
fps.stop()
print("[INFO]Elapsed time: {:.2f}".format(fps.elapsed()))
print("[INFO]Approx. FPS:  {:.2f}".format(fps.fps()))


4.5: TESTING APPROACH
Software testing has the power to point out all the defects and flaws during development. . Different kinds of testing allow us to catch bugs that are visible only during runtime.
The purpose of machine learning testing is to ensure that this learned logic will remain consistent, no matter how many times we call the program.
Functional Testing
It is a type of software testing that validates the software system against the functional requirements/specifications. 
The purpose of Functional tests is to test each function of the software application, by providing appropriate input, verifying the output against the Functional requirements.

 
Functional testing mainly involves;
Black-box testing of machine learning (ML) models refers to testing with no knowledge about the internal details of the model, such as the algorithm used to create it and the features in it. The main objective of black-box testing is to ensure the quality of the models in a sustained manner.
Unit tests. The program is broken down into blocks, and each element (unit) is tested separately
It involves testing individual units of the source code, such as functions, methods, and class to ascertain that they meet the requirements and have expected results.
Each piece of code has been tested individually and results executed.
Regression tests. They cover already tested software to see if it doesn’t suddenly break and also ensures quality of the user experience along with the new changes.   
 Integration tests
These tests aim to determine whether modules that have been developed separately work as expected when brought together. In terms of a data pipeline, these can check that:
•	The data cleaning process results in a dataset appropriate for the model
•	The model training can handle the data provided to it and outputs results (ensuring that code can be refactored in the future)
•	The data is consumable by the model (a label exists for every input; the types of the data are accepted by the type of model chosen)
We are able to refactor our code in the future, without breaking the end-to-end functionality.
4.6: MODIFICATION AND IMPROVEMENT
The system has room for improvement in making able to calculate the distance between individuals who are interlocking one another in the video stream.

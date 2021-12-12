# Driver_Drowsiness
The project is about the indication of driver's drowsiness system. It observes the drivers behaviour and buzz the alarm if the driver is sleepy while driving. With the implimentation of this project, we could avoid the road traffic colleasion to certain level. 

In this Python project, I used be using OpenCV for gathering the images from webcam and feed them into a Deep Learning CNN trained model which will classify whether the person’s eyes are ‘Open’ or ‘Closed’. The approach we will be using for this Python project is as follows :

Step 1 – Take image as input from a camera.

Step 2 – Detect the face in the image and create a Region of Interest (ROI).

Step 3 – Detect the eyes from ROI and feed it to the classifier.

Step 4 – Classifier will categorize whether eyes are open or closed.

Step 5 – Calculate score to check whether the person is drowsy.

# Data Collection
I downloaded the data(images) from kaggle. There are two classes such as 'Closed' and 'Open'. Since there were less number of images, I used data augumentation while training the model. the images are in color however I converted them into GRAYSCALE before I feed them into the model.

# CNN Model
and used CNN Network to build and train a model. The architecture of the final model goes as below, 
- 32, 3x3, 'relu', (150,150,1)
- maxpooling2d(2,2)
- 32, 3x3, 'relu'
- maxpooling2d(2,2)
- 32, 3x3, 'relu'
- maxpooling2d(2,2)
- 1, 'sigmoid'

With this model, I achieved 100% test accuracy. 
Before this, I worked on different architecture and with different Optimizers. Here I have attached few of their performace visualisation below. 
![image](https://user-images.githubusercontent.com/75533233/145724573-d4946870-1431-4529-8ec7-836ff4097182.png)
![image](https://user-images.githubusercontent.com/75533233/145724587-e0149421-d09a-4560-9079-32157adbbe0d.png)
![image](https://user-images.githubusercontent.com/75533233/145724590-7b0438b0-bdb1-4344-bdf1-ccbe196216c4.png)
![image](https://user-images.githubusercontent.com/75533233/145724594-24576cfb-7bf6-4f41-8073-8c42cdaf3c41.png)

The final model's results with Adam as optimizer are below 
![image](https://user-images.githubusercontent.com/75533233/145724604-e99a6f11-8480-4fb2-bf1d-22e49c2cf897.png)

After I used Early stop function
![image](https://user-images.githubusercontent.com/75533233/145724610-90e13b24-74b5-4425-9b4e-c78db788b736.png)

# Pre-build models
- "haarcascade_frontalface_alt.xml", "haarcascade_lefteye_2splits.xml" and "haarcascade_righteye_2splits.xml" files that are needed to detect objects from the image. In our case, we are detecting the face and eyes of the person.
- The “Drowsiness_CNN_Model_tf1.h5” which was trained on Convolutional Neural Networks.
- There is an audio clip “alarm.wav” which is played when the person is feeling drowsy.
- “Live Prediction.ipynb” is the main file of our project. To start the detection procedure, we have to run this file.
- "Driver_Drowsiness_CNN_Network.ipynb" file is the CNN architecture and model training code of the final model.

# Result 
The below are the output of the project 

![Open](https://user-images.githubusercontent.com/75533233/145725038-9476e7a4-4232-42bc-b0e8-1e63dccd097e.png)

![Closed](https://user-images.githubusercontent.com/75533233/145725048-71c19702-3d52-44d0-a77d-ffaa78fb5b40.png)




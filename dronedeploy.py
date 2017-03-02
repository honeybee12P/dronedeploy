import cv2
import numpy as np
from scipy import linalg
from pylab import *

###########################################SUPPORTING FUNCTIONS##################################################

#this function sets the camera (iphone6) parameters
def camera_calibration(sizes):  
    row, col = sizes
    x = 2555 * col / 2592
    y = 2586 * row / 1936
    camera_parameters = np.diag([x, y, 1])
    camera_parameters[0, 2] = 0.5 * col
    camera_parameters[1, 2] = 0.5 * row
    return camera_parameters

#this function helps us draw cube which is considered as camera in this case.
def cube_points(c, wid): 
    
    cube_parameters = []
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] - wid])
    cube_parameters.append([c[0] - wid, c[1] + wid, c[2] - wid])
    cube_parameters.append([c[0] + wid, c[1] + wid, c[2] - wid])
    cube_parameters.append([c[0] + wid, c[1] - wid, c[2] - wid])
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] - wid])
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] + wid])
    cube_parameters.append([c[0] - wid, c[1] + wid, c[2] + wid])
    cube_parameters.append([c[0] + wid, c[1] + wid, c[2] + wid])
    cube_parameters.append([c[0] + wid, c[1] - wid, c[2] + wid])
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] + wid])
    
    cube_parameters.append([c[0] - wid, c[1] - wid, c[2] + wid])
    cube_parameters.append([c[0] - wid, c[1] + wid, c[2] + wid])
    cube_parameters.append([c[0] - wid, c[1] + wid, c[2] - wid])
    cube_parameters.append([c[0] + wid, c[1] + wid, c[2] - wid])
    cube_parameters.append([c[0] + wid, c[1] + wid, c[2] + wid])
    cube_parameters.append([c[0] + wid, c[1] - wid, c[2] + wid])
    cube_parameters.append([c[0] + wid, c[1] - wid, c[2] - wid])
    return np.array(cube_parameters).T


#converts points to plot
def homogeneous_coordinates(points):  
    
    return np.vstack((points, np.ones((1, points.shape[1]))))


class Camera(object):

    def __init__(self, P):

        self.P = P
        self.K = None
        self.R = None
        self.t = None
        self.c = None

    #calculates the matrix of the camera position
    def project(self, X): 
        
        x = np.dot(self.P, X)
    
        for i in range(3):
            x[i] = np.divide(x[i],x[2])
        return x


###########################################READING THE TWO IMAGES########################################################


reference_image = cv2.imread('images/pattern.jpg', 0)  #pattern image
main_image = cv2.imread('/Users/kruthikavishwanath/Desktop/dronedeploy/images/img7.JPG', 0)  #pattern on carpet



###########################################FINDING THE MATCH BETWEEN TWO IMAGES###########################################

#sift object
sift = cv2.xfeatures2d.SIFT_create() 

#detects the key points in the reference_image
keyPoints1, descriptors1 = sift.detectAndCompute(reference_image, None) 

#detects the key points in the main_image
keyPoints2, descriptors2 = sift.detectAndCompute(main_image, None) 

#brute force matching - descriptor of feature from reference_image is matched with descriptors in main_image using distance calculation
bf_object = cv2.BFMatcher() #bf_object is the object being created

#both the descriptors from the two images are matched
matches = bf_object.knnMatch(descriptors1, descriptors2, k=2) 

#best matches from the two images are calculated.
goodMatches = []

for m, n in matches:
    if m.distance < 0.7 * n.distance:
        goodMatches.append(m)


MIN_MATCH_COUNT = 10

if len(goodMatches) > MIN_MATCH_COUNT:
    
    #finds the positions of good points
    source_points = np.float32([keyPoints1[m.queryIdx].pt for m in goodMatches]).reshape(-1, 1, 2)
    destination_points = np.float32([keyPoints2[m.trainIdx].pt for m in goodMatches]).reshape(-1, 1, 2)

    
    #returns a mask which specifies the inlier and outlier points of the images.
    M, mask = cv2.findHomography(source_points, destination_points, method=cv2.RANSAC, ransacReprojThreshold=5.0) 
   

    #this creates the thin silver line around the pattern on the carpet
    h, w = reference_image.shape
    corners = np.float32([[0, 0], [0, h - 1], [w - 1, h - 1], [w - 1, 0]]).reshape(-1, 1, 2)

    #perspective transforms helps one find the pattern present in image 1 which is also found in another image. 
    transformedCorners = cv2.perspectiveTransform(corners, M)

    #polylines helps in circling the pattern we are finding the carpet image  
    main_image = cv2.polylines(main_image, [np.int32(transformedCorners)], True, (255, 255, 255), 2, cv2.LINE_AA) 

else:
    print("Insufficient matches found")



###########################################ESTIMATING THE POSE OF THE CAMERA AND DRAWING THE CAMERA(CUBE)###########################################

#parameters of camera(iphone6) is set
camera_parameters = camera_calibration((747, 1000)) 

#parameters (Length, breadth & height) of the cube (representing camera here) is set 
box = cube_points([0, 0, 0.1], 0.1) 

#camera matrix is found by using mask i.e., good matches (M) from homography and position of the pattern image
camera_position1 = Camera(np.hstack((camera_parameters, np.dot(camera_parameters, np.array([[0], [0], [-1]])))))
camera_position2 = Camera(np.dot(M, camera_position1.P))
A = np.dot(linalg.inv(camera_parameters), camera_position2.P[:, :3])
A = np.array([A[:, 0], A[:, 1], np.cross(A[:, 0], A[:, 1])]).T
camera_position2.P[:, :3] = np.dot(camera_parameters, A)

#coordinates of the camera (how the camera is kept while taking the picture) is found here by using the camera matrix.
camera_cube = camera_position2.project(homogeneous_coordinates(box))  


####################################PLOTTING THE MATCHES ALONG WITH CUBE(CAMERA) ON THE CARPET IMAGE##############################################

pattern = np.array(reference_image)
carpet_image = np.array(main_image)

imshow(carpet_image)

plot(camera_cube[0,:],camera_cube[1,:],linewidth=6)

axis('off')

show()
## Advanced Lane Finding Project


---

[//]: # (Image References)

[image1]: ./chessboard_images/chessboard_image1.jpg "Chessboard image 1"
[image2]: ./chessboard_images/chessboard_image2.jpg "Chessboard image 2"
[image3]: ./chessboard_images/chessboard_image17.jpg "Chessboard image 17"
[image4]: ./undistorted_chessboard_images/undistorted_chessboard_image6.jpg "Undistorted_ Chessboard image 6"
[image5]: ./undistorted_chessboard_images/undistorted_chessboard_image8.jpg "Undistorted_Chessboard image 8"
[image6]: ./undistorted_chessboard_images/undistorted_chessboard_image18.jpg "Undistorted_Chessboard image 17"
[image7]: ./output_images/combo_orig_undist_cb_image.jpg "Combo Chessboard image"
[image8]: ./undistorted_images/undist_image2.jpg "Undistorted image 1"
[image9]: ./undistorted_images/undist_image4.jpg "Undistorted image 2"
[image10]: ./output_images/combo_perspective_image.jpg "Combo Perspective image"
[video1]: ./output_videos/output_video.mp4 "Video"

---

### Writeup / README

From a very high level the strategy i followed for this project was to dilligently follow the individual project videos and try out all the various variations using the test images using my own ipython notebooks. All the notebooks are checked into the project repository. However only 2 of them are actual required for this project submission itself.

The 2 important ipython noetbooks for this project are

```sh
P4_Camera_Calibration.ipynb
```

and

```sh
P4_Tranform_undistorted_images.ipynb
```

### Camera Calibration

The whole camera calibration is done in a separate ipthon notebook. This notebook is called **P4_Camera_Calibration.ipynb**

First an foremost i have taken note of the fact that the images are9x6 chessboard images and i ahve updated my code appropriatley.

The code is below. As can be seen, i first use open cv to find the chessboard corners.
If these are present i proceed to drawt the image using chessboard corners. I msut make a note of the fact that not all images have chessboard corners being detected. There were a total of **seventeen** images that had chessboard corners.

```sh
nx = 9
ny = 6
count = 0

#prepare object points
objpoints = [] #3D points in real world space
imgpoints = [] #2D points in image plane

objp = np.zeros((ny * nx, 3), np.float32)
objp[:,:2] = np.mgrid[0:nx,0:ny].T.reshape(-1, 2) #x and y co-ordinates


for i in range(len(images_to_load)):
    img = images_to_load[i]
    
    #Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    #Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (nx, ny), None)

    #If found, draw corners
    if ret == True:
        count =  count + 1
        imgpoints.append(corners)
        objpoints.append(objp)
        #Draw and display the corners
        cv2.drawChessboardCorners(img, (nx, ny), corners, ret)
        filename = './chessboard_images/chessboard_image'+str(i)+'.jpg'
        cv2.imwrite(filename, img)
        plt.imshow(img)
        plt.show()
```

Some example images can be seen below

![alt text][image1]


![alt text][image2]


![alt text][image3]



Before i proceed to undistort these images, I save the appropriate data using pickle.

I save using the following sequence of code.

```sh
points_pickle = {}
points_pickle["objpoints"] = objpoints
points_pickle["imgpoints"] = imgpoints

pickle.dump(points_pickle, open("camera_calibration_points.p", "wb" ) )
```

I then obtain obtain these saved points using the sequence of code below.

```sh
dist_pickle = pickle.load( open("camera_calibration_points.p", "rb" ) )
objpts = dist_pickle["objpoints"]
imgpts = dist_pickle["imgpoints"]
```

I then proceed to undistort the images using the calibrateCamera function of open cv using the function below.

```sh
def cal_undistort(image, obj_points, img_points):
    #Use cv2.calibrateCamera() and cv2.undistort()
    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image.shape[1::-1], None, None)
    dst = cv2.undistort(image, mtx, dist, None, mtx)
    return dst
```


I also save some of the important data using pickle. Thsi can be seen below.

```sh
undist_pickle = {}
undist_pickle["mtx"] = mtx
undist_pickle["dist"] = dist
pickle.dump(undist_pickle, open("mtx_dist_pickle.p", "wb" ))
```

Some undistorted example images can be seen below

![alt text][image4]


![alt text][image5]


![alt text][image6]


Below we can see a side by side image of a chessboard image with and without distortion after having applied the undistortion.

![alt text][image7]


### Pipeline (single images)

#### 1. Distortion Corrected Test Images.

I read in the eight test images and undistort all eight of them using opencv. Belwo i present 2 examples of such undistorted images.

![alt text][image8]


![alt text][image9]


#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.

From this section onwards, all of the pertinent code can be seen in the project notebook.
```sh
P4_Tranform_undistorted_images.ipynb
```

I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines # through # in `another_file.py`).  Here's an example of my output for this step.  (note: this is not actually from one of the test images)



#### 3. Perspective transform

The code for my perspective transform is below.

```python
#Grab the image shape
img_size = (image_t.shape[1], image_t.shape[0])
    
left_upper_point  = [580,460]
right_upper_point = [700,460]
left_lower_point  = [260,680]
right_lower_point = [1050,680]

src = np.float32([left_upper_point, left_lower_point, right_upper_point, right_lower_point])
dst = np.float32([[200,0], [200,680], [1000,0], [1000,680]])

#Given src and dst points, calculate the perspective transform matrix
M = cv2.getPerspectiveTransform(src, dst)
#Minv is required later in the project
Minv = np.linalg.inv(M)

#Warp the image using OpenCV warpPerspective()
warped = cv2.warpPerspective(image_t, M, img_size, flags=cv2.INTER_NEAREST)

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(24, 9))
f.tight_layout()
ax1.imshow(image_t)
ax1.set_title('Undistorted Image', fontsize=50)
ax2.imshow(warped)
ax2.set_title('Warped Image.', fontsize=50)
plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
plt.show()
```

This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 580,  460     | 200,  0       | 
| 700,  460     | 200,  680     |
| 1050, 680     | 1000, 680     |
| 260,  680     | 1000, 0       |


An example is given below

![alt text][image10]


#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

Then I did some other stuff and fit my lane lines with a 2nd order polynomial kinda like this:



#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines # through # in my code in `my_other_file.py`

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines # through # in my code in `yet_another_file.py` in the function `map_lane()`.  Here is an example of my result on a test image:



---

### Pipeline (video)

I am providing a link to the project output video below. This is also embedded into the ipython notebook. 

```sh
P4_Tranform_undistorted_images.ipynb
```

Here's a [link to my video result](./output_videos/output_video.mp4)

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?


Some of the issues faced during the project are below.

* I had a major typo in my project which crated bad output and took a lot fo time to rectify. I explain this below.

I have this section of code in my project.

```sh
hls_binary = hls_select(undist_image, thresh=(150, 255))

#i then have the following sequence to combine thresholds.
combined[((gradx == 1) & (grady == 1)) | (hls_binary == 1)] = 1
```

Instead of **(hls_binary == 1)** i had **(hls_select == 1)**

This caused some major issues for me.


* The perspective transform is a very critical part of the project. It is very impotrant to get this as **right** as possible. I had to spend a lot of time fine tuning this and trying out my project video with this perspective transform.

* Combining Thresholds in time consuming but it is a fun experience.



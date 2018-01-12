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
[image11]: ./output_images/combo_thresh_image1.jpg "Combo Threshold image"
[image12]: ./output_images/combo_thresh_image2.jpg "Combo Threshold image"
[image13]: ./output_images/combo_thresh_image3.jpg "Combo Threshold image"
[image14]: ./binary_warped_images/binary_warped_line_image1.jpg "Binary Warped image 1"
[image15]: ./binary_warped_images/binary_warped_line_image2.jpg "Binary Warped image 2"
[image16]: ./output_images/output_image2.jpg "Output image 2"
[image17]: ./output_images/output_image3.jpg "Output image 3"
[image18]: ./output_images/output_image4.jpg "Output image 4"
[image19]: ./output_images/output_image5.jpg "Output image 5"
[image20]: ./output_images/output_image6.jpg "Output image 6"
[image21]: ./output_images/output_image7.jpg "Output image 7"
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

I read in the eight test images and undistort all eight of them using opencv. Below i present 2 examples of such undistorted images.


I load the previously saved **mtx** and **dist** information. I show this step below.

```sh
undist_pickle = {}

undist_pickle = pickle.load(open("mtx_dist_pickle.p", "rb"))

mtx = undist_pickle["mtx"]
dist = undist_pickle["dist"]
```


![alt text][image8]


![alt text][image9]


#### 2.  Thresholded Binary Image

From this section onwards, all of the pertinent code can be seen in the project notebook.
```sh
P4_Tranform_undistorted_images.ipynb
```

```sh
def get_undistorted_combined_warped_binary_image(img):
    undist_image = cv2.undistort(img, mtx, dist, None, mtx)
    
    #Apply each of the thresholding functions
    gradx = abs_sobel_thresh(undist_image, orient='x', sobel_kernel=3, thresh=(20, 120))
    grady = abs_sobel_thresh(undist_image, orient='y', sobel_kernel=3, thresh=(20, 120))
    
    hls_binary = hls_select(undist_image, thresh=(150, 255))
    
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | (hls_binary == 1)] = 1

    combined_warped = cv2.warpPerspective(combined, M, (1280, 720), flags=cv2.INTER_NEAREST)
    
    return combined_warped
```

In my project i have a function called get_undistorted_combined_warped_binary_image which first undistorts the test images.

After this i obtain the absolute sobel thersholds in both the x and y direction. In both the cases i use a kernel size of 3 and threshold range is 20 to 120.

I then use the s channel image with a threshold between 150 to 255.

At this stage of the project I spent a lot of time trying various combinations and finally settled for the combination shown above.

Below i show some examples.

![alt text][image11]


![alt text][image12]


![alt text][image13]


#### 3. Perspective transform

The code for my perspective transform is below. 

The source and destination points involved a lot of experimentation.

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


#### 4.  Identifying Lane Line Pixels

I did leverage a lot of the starter code for obtaining the lane lines as shown in the project videos. I did have my own function to make the code moer modular.

I have 2 separate function to detect lane lines.

The first one is below and is used to find the lane lines for the test images.

```sh
def finding_the_lines(binary_warped, windows = 9, margin = 100, minpix = 50):
```

I have another more "advanced" lane lines function that i use in the project video output.

```sh
def advanced_find_lines(binary_warped, left_fit, right_fit):
```

In the project test video both the functions are used to detect lane lines.


The first function above takes in a binary warped image and tkaes a histogram of the bottom half of the image. I then find the peak of the left and right halves of the histogram. These will be the starting point for the left and right lines. I then proceed to perform a sliding window search and choose the number of windows and the height of the windows. I then identify the x and y positions of all nonzero pixels in the image and appropriately set the margins. Then i have a for loop to step through the window one by one. I find the left and right indices and append them to lists. I extract left and right line pixel positions and finally fit a second order polynomial to each. This function iself is enough to find lanes in the test images.

This needs to be enhanced for the video since we do not want to go through the entire process of window search for each and every frame in the video. My **advanced_find_lines** function takes in a binary warped image. It extracts left and right line pixel positions and fits in a second order polynomial.

As an intermediate step to actually visualize the lines in the bianry warped image i have another function called **visualize_the_lines** which takes in a binary warped image as input.
 
```sh
def visualize_the_lines(binary_warped):
```

the function above generates x and y values for plotting. Then creates an image to draw on and an image to show the selection window. Then generate a polygon to illustrate the search window area and recast the x and y points into usable format for cv2.fillPoly(). Then draw the lane onto the warped blank image. Examples can be seen below.


![alt text][image14]


![alt text][image15]


#### 5. Calculating the Radius of Curvature of the Lane and the Position of the Vehicle with respect to center.

I did this in lines through the **obtain_curvature** function.

The function header is shown below. This is called immediately after finding the right and left lanes.

```sh
def obtain_curvature(binary_warped, left_fit, right_fit):
```
The project helper videos had the following important tip that i picked up.

```sh
#Define conversions in x and y from pixels space to meters
ym_per_pix = 30/720 #meters per pixel in y dimension
xm_per_pix = 3.7/700 #meters per pixel in x dimension
y_eval = np.max(ploty)
```
I fit new polynomials to x,y in world space and then calculate the new radii of curvature in meters.



#### 6. Output Images

I implemented this step using the **draw_image** function.

After undistorting the images and creating a combined thresholds image, i proceed to find lanes and get curvature with the warped image. i then draw the lane onto the warped blank image and warp the blank back to original image space using inverse perspective matrix (Minv). I print out three pieces of information on each image.

This is shown below.

```sh
cv2.putText(result, 'Left radius: {:.0f} m'.format(left_curverad), (50, 50), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
cv2.putText(result, 'Right radius: {:.0f} m'.format(right_curverad), (50, 100), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2) 
    
if (center < 0):
   cv2.putText(result, 'vehicle is left of center by: {:.2f} m'.format(-center), (50, 150),    cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2) 
else:
    cv2.putText(result, 'vehicle is right of center by: {:.2f} m'.format(center), (50, 150), cv2.FONT_HERSHEY_DUPLEX, 1, (255, 255, 255), 2)
```


Below I show the output of all test images.


![alt text][image16]


![alt text][image17]


![alt text][image18]


![alt text][image19]


![alt text][image20]


![alt text][image21]

---

### Pipeline (video)

The following sequence of code preocess the video.

```sh
def process_image(image):
    return pipeline(image)
    
output_video = './output_videos/output_video.mp4'
clip = VideoFileClip('./project_video.mp4')
output_clip = clip.fl_image(process_image)
%time output_clip.write_videofile(output_video, audio=False)
```

I handle the video a little different compared to the static test images. The **pipeline** is where the each frame of the video is processed. In this after undistorting the images and creating a combined thresholds image, i proceed to find lanes and get curvature with the warped image. i then draw the lane onto the warped blank image and warp the blank back to original image space using inverse perspective matrix (Minv).


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

* I can see that the lines get a little wobbly when the color of the road surace changes. In one case, thsi is probably exacerbated by the presence of tress. These are all standard cases in the real world. The pipeline will need to be enhanced to take care of this eventuality.

* My pipeline definitely needs to be enhanced with better tracking of lane lines and a smarter algorithm than what I have now. I will need to ahve multiple class objects for both left and right lanes.



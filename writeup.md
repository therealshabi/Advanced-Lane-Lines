## Writeup Template

This project was aimed for detecting Curved Lane Lines of the road under varied lightning conditions using techniques like Perspective Transform, Color Thresholding etc..

---

**Advanced Lane Finding Project**

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[undistort]: ./project_images/undistort.png "Undistorted"
[original_image]: ./project_images/original_image.png "Original Image"
[undistort_test]: ./project_images/undistort_test.png "Undistort test"
[sobel_binary]: ./project_images/sobel_thresholding.png "Sobel Thresholding"
[color_binary]: ./project_images/color_binary.png "Color Binary"
[combined_binary]: ./project_images/combined_binary.png "Combined Binary"
[masked_image]: ./project_images/masked_image.png "Masked Image"
[perspective_transformed_image]: ./project_images/perspective_transformed_image.png "Warped Image"
[lane_poly_fit]: ./project_images/lane_poly_fit.png "Lane poly fit Image"
[curvature]: ./project_images/curvature.png "Curvature Image"
[final_output]: ./project_images/result.png "Final Image"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points

### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---

### Advanced Lane Detection Pipeline
---
The pipeline for detecting lane lines on the road consists of a number of steps.

### Camera Calibration

The code for this step is contained in the first code cell of the IPython notebook located in "./examples/example.ipynb" (or in lines # through # of the file called `some_file.py`).  

- `
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function. I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result for the chess board camera part: 

![alt text][undistort]

### Pipeline (single images)

**`pipeline(image)`**: This is the name of my pipeline function which includes following lines and steps:-

- `original_image = np.copy(image)`: First of all I made a copy of the original image to apply all the transformations on the copy rather than the original image.

	![alt text][original_image]

- `undist_image = undistort(original_image)`: This is a method used for undistorting the test images. So here I used the undistortion and calibration coefficients obtained while undistorting and calibrating camera through 20 different chessboard images of varied orientation and subsequently applied the same on my test image to undistort the image.

	![alt text][undistort_test]
	
- `combined_binary = binary_thresholding(undist_image)`: I used a combination of color and gradient thresholds to generate a binary image. For that purpose I used a method called `binary_thresholding` which took the undistorted image as it's parameter and apply thresholding into it which includes:

	**`sobel_thresholding`**: I took the gradient in the x direction with the threshold most suitable for my test image.

	![alt text][sobel_binary]

	**`color_thresholding`**: After choosing numerous threshold values for different image channel divisions, I found that the Saturation and Light factor of an HSV image along with the Red component of the RGB image are the most suitable deciding factors for a clear line detection, so I took suitable thresholds for each of them and obtained a nice looking color binary.

	![alt text][color_binary]

	In the end of this step, I combined the activated binaries of both the gradient and color thresholds and produces a combined_binary as an output for the next step.

	![alt text][combined_binary]

- `masked_image = masking(combined_binary)`: Post obtaining the binary, I noticed still a lot of unwanted curves were detected, so I took a mask of the image to the part only required for lane detection.

	![alt text][masked_image]

- `perspective_transform_image = perspectiveTransform(masked_image)`: The code to my perspective transform was to obtain a bird's eye view of the lane so as to estimate the curves more efficiently. The method took the masked image and just mapped the source points in the masked image to the destination point of the required image. Here's the code for that:

```python
def perspectiveTransform(img):
    #set source and destination points
    src = np.float32([(200,720),(1080,720),(700,460),(580,460)])
    dst = np.float32([(200,720),(1080,720),(1080,0),(200,0)])
    
    #apply perspective transform
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, (img.shape[1],img.shape[0]))
    
    return warped
```

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 200, 720      | 200, 720      | 
| 1080, 720     | 1080, 720     |
| 700, 460      | 1080, 0       |
| 580, 460      | 200, 0        |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][perspective_transformed_image]

**Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?**

Post that I used sliding window technique to determine the fitting polynomial. For this first I divided my warped image to 9 windows and for each window calculated a histogram, the 2 histogram location with the highest peaks give a fair idea about the lanes, and we trace through each of this histogram throughout the sliding window to obtain the left and the right lines. Subsequently, we store the left and right line indices and draw a second order polynomial (coz it's a curve) over the image roughly estimating it's curve.

![alt text][lane_poly_fit]

The `sliding_window_from_prior` function performs basically the same task, but it reduces much trouble of the procedure by utilizing a past fit (from a prior video frame, for instance) and hunting down path pixels inside a specific scope of that fit. The picture beneath shows this - the green shaded region is the range from the past fit, and the yellow lines and red and blue pixels are from the present picture:

- `draw_radius_curvature(warped,img,left_fitx,right_fitx)`: This take the warped image as the parameter and calculate the average radius of curvature of the lines detected in the image frame. Here's the code part of that: 

```python
curverad = ((1 + (2*fit_cr[0]*y_eval*ym_per_pix + fit_cr[1])**2)**1.5) / np.absolute(2*fit_cr[0])
```
In this example, fit_cr[0] is the first coefficient (the y-squared coefficient) of the second order polynomial fit, and fit_cr[1] is the second (y) coefficient. y_eval is the y position within the image upon which the curvature calculation is based. Here we chose the maximum y-value, corresponding to the bottom of the image. ym_per_pix is the factor used for converting from pixels to meters. This conversion was also used to generate a new fit with coefficients in terms of meters.

We also calculated the center offset or the position of the vehicle with respect to the center of the lane with the help of the following code:

```python
lane_center = (left_fitx[-1] + right_fitx[-1]) // 2
car_center = img.shape[1]/2
center_offset = (lane_center - car_center) * x_m_per_pix
```
Here left_fitx and right_fitx are the left and right x intercepts of left fit and left fit respectively calculated from sliding window step.

Additionally, I also calculated the line accuracy of the both the curve fits by calculating the average width between the lanes and comparing against suitable threshold values, this way the error was reduced considerably.

![alt text][curvature]

**An example image of my result plotted back down onto the road such that the lane area is identified clearly**
Finally I mapped the radius of curvature outline to the original image by applying reverse perspective transform from destination to source and displayed the curvature of the road along with the offset. Here is the resultant image:

![alt text][final_output]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my video result](./output_images/output_video/project_video_output.mp4)

---
### Identify potential shortcomings with your current pipeline

While doing this project, I discovered a number of shortcomings with my current algorithm which includes:-

- When the lane lines are irregular like that in the challenge videos the detection went haywire.

- Under varying lightning conditions the algorithm failed to detect the line let alone the curvature of the line.

- Since I took a large field of view, the algorithm really struggled to cover the immediate or blind curves.

---
### Suggest possible improvements to your pipeline

I've though about some possible improvements with my approach which includes:

-  A heuristic to obtain the dynamic color thresholding of the image which could possibly result in better detection of curves along varied frames.

- Usage of the other color channels like HSV and LAB can also result in better thresholding of images.

- Masking post the perspective transform can remove any small unwanted noise in the image.
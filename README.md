# Tennis-Player-Distance-Measurement
Measure the distances travelled by each player on the court, during a course of tennis match from a given clip/video. 

Problem Statement 1: Measure the distance travelled by both the players individually while playing on the court during the course of this clip.

SOLUTION:
1. The video clip shared originally had 120 FPS. The complete video had frames from multiple camera views installed on the premises (each for a different use case, like, focusing on the court, focusing on players and other use cases). So, the focus was to extract the video clips (or frames) where the focus was on the game (or the court).
2. A court image 'court_reference.jpg' was picked up randomly from a few of the starting frames, which would treat as a reference to compare court images.
3. The script "split_video.py", downsamples the 120 fps video to 60 fps (to reduce the computation expense) and extracts the frames focusing on the court and saves it to 'selected_frames_60fps' directory. To compare to frames with the reference court image, the Imagehash AverageHash has been used, with a threshold of 20 bits i.e. if Hamming distance between 2 images' hash is more than 20bits, then the images are dissimilar.
4. Using the script "court_detector.py", the court coordinates are detected in the reference court image, by thresholding the white pixels, detecting lines using  Hough Transformation, merging the overlapping lines and a few other steps. The convex points, outermost points of the court references are pickled for future reference.
5. The selected court frames are binned into multiple smaller clips for better interpretability, using the difference between the frame numbers.
6. "measure_player_distances.py" - The approach to detect and measure the distance travelled by each player, YOLOv5 has been used to detect the player coordinates. Calculating the Euclidean distance between the player coordinates from two positions can give us the distance travelled by each player within two frames, in pixel units. BUT.... the selected court frames and the selected camera view, suffer from the problem of a perspective view. This means that the 10 pixels unit distance covered by the 1st player is not equal to the distance travelled by the 2nd player in the 10 pixels unit. To solve this problem, "warpPerspective" from the OpenCV library has been used twice, for an effective transformation of perspective court frames to bird-view frames, and accurate measurement of distance travelled. Refer to the "warped_court.jpg" image to see the final output.
7. Using the bird-view frames, we have a fair estimation of what does 1-pixel unit distance represents in metres on the x-axis and y-axis. These values are used to scale the displacement in the x and y-axis and estimate the distance covered by each player.
8. RESULT: Estimated distance travelled is 697 metres (by Player 1 i.e. on the farther side of the camera) and 538 metres (by Player 2 i.e. near to the camera) for 1 round of tennis match (roughly 4 minutes).


Problem Statement 2: Detect the number of times the players have served
SOLUTION:
1. Most ideal way would be to train an LR-CNN network and perform Activity Recognition on the sequence of frames, with each frame focusing on an individual player, but the challenging part is the training data required to do that. So, after observing and analyzing the video a frequent pattern was observed. Pattern being, after each serve breaks or when any player gains a point, the camera angle changes, creating a different view than what is shown in "court_reference.jpg". 
2. This break/discontinuity can be easily counted in the clips that we have already identified. We had already broken our complete video into smaller clips, with each clip having frames focusing on the tennis court.
3. To get an estimate of the number of actual/true serves made by players, we can count the number of clips with at least 60 frames, or clips with a duration of at least 1 second. In the video shared with the assignment, we have identified a total of 29 clips, and out of them, 25 clips are long enough (duration > 1 sec) to logically capture a serve.

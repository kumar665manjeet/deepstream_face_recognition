This folder contains face recognition using deepstream python application with custum input file format like .h264 .mp4 file as input and also for output file as EGL sink or .mp4 format or sink as rtsp at custum port.

Some file also give the face crop in specified directory name.

#### To run these application follow these procedure:

* deepstream-face-mp4-mp4.py file:///home/app/11.mp4 frame

* deepstream-face-rtsp.py   <<v4l2-device-path>> e.g /dev/video0 

* deepstream-face-mp4-rtsp.py file:///home/app/11.mp4   <face crop file name>

* deepstream-face-mp4-out.py <.h264 file> <face crop file name>

* deepstream-face-egl-out.py  <uri1> [uri2] ... [uriN] <folder to save frames>



** Change the primery and secondary comnfigration file path and model file path

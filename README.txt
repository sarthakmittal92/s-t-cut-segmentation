Dependencies:
python 3.5+
libopencv-dev
python3-tk

Other packages in requirements.txt
Install using: pip install -r requirements

virtualenv can be used.

Usage:
run using: python3 ./code/Segmentation.py -i <path-to-input>
eg. 'python3 ./code/Segmentation.py -i ./data/deer.png'

Details:
GUI shows up to mark foreground and background
Toggle using 'b' for background and 'o' for foreground
Press ESC after making the scribbles (sample scribbled images added in data folder)
Output will be stored in results directory (as out.png)
# The_Pose

##installing Docker
Please use the following command to install docker on smaller devices. So the rest of the code can be implemented in the docker container: 
'''
curl -sSL https://get.docker.com | sh
'''


The code here is based on the MPII human pose dataset and the model is going to detect human pose for an edge device.


## Reading the COCO dataset

In order to read the COCO imageset annotations for the key_point detection, we installed the COCO API.
For this installation first the execute "make" under cocoAPI/PythonAPI and then run the command "Sudo setup.py install". This way the library os added to the python path and is reached outside of the library.




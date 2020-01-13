# AMLSassign--ment19_20
UCL_AMLS_I

## Set up

#### 1 - Install dev prereqs (use equivalent linux or windows package management)
        brew install python3.6
        brew install virtualenv
        
Example for Windows when not using any package management software and limited or no admin rights:

- Install python 3.6 from https://conda.io/miniconda.html and make sure you select 'add to path' when prompted byt the 
installation wizard
- If you failed to select 'add to path' during the installation: Go to Control Panel - User Accounts - Change my 
environment variables. Under User variables find 'Path' and click 'Edit'. Then click new and paste the path to where 
Miniconda was installed - the path can be found by opening Anaconda prompt and running:

        echo %PATH%
        
- Miniconda comes with virtual environment management so ignore the brew installations above, proceed to Section 2 and
follow Windows specific requirements.

- The project requires dlib - install it into your environment following the instructions at https://www.pyimagesearch.com/2018/01/22/install-dlib-easy-complete-guide/. 


#### 2 - Set up a Python virtual environment (from the root of the AMLS directory)
        virtualenv -p python3.6 AMLS-venv
        source AMLS-venv/bin/activate


For Windows, open cmd and run the following commands, hitting enter after each line and waiting for it to execute:  (** path-to-script ** is where you have downloaded this repository)

        cd "** path-to-script **"
        conda create -n AMLS-venv python=3.6
        activate ASMLS-venv


#### 3 - Install required python dependencies into the virtual env
        pip install -r requirements.txt

#### 4 - Run the AMLS 

The data file and the build ML models should be downloaded from the following Google Drive link: https://drive.google.com/drive/folders/17ANTXC4wyGfZ9h0fgqzYBspVatDDiqxu
In the shared drive, there are 4 folders, Task A1, Task A2, Task B1 and Task B2. Each folder contains the task relative model and data files used in tarining and testing the model.
#### 5 - Run the gender_classifier.py, emotion_detector.py, eye_color_classifier.py and face_shape_classifier.py scripts
 separately (from inside the virtual env)
Giving: 


        postional arguments:
        -i --img-dir  The path to the directory where the 
                     images are
        
        -l --labels-file  The path to the csv labels file
        
        -fci --face-shape-index  The index of the face shape column 
                                 in the labels.csv file
        
        -eci --eye-color-index  The index of the eye color column 
                                in the labels.csv file
        
        -gi --gender-index  The index of the gender column 
                            in the labels.csv file
        
        -ei --emotion-index  The index of the smiling column 
                             in the labels.csv file
        
        additional positional arguments for gender_classifier.py and emotion_detector.py: 
        
        -s --landmarks-file The path to the face landmarks file
        
        -pd --preprocessed-data-file  The path to the preprocessed image data file
        
        
      

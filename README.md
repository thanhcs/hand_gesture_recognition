project 4 description
=====================

![Alt text](Capture.PNG?raw=true "Energy Images")

Our project 4 implements hand gesture recognition using energy images and SVM classification.

There were several prototypes leading to final implementation, but the final code can be found in *trainer.py*.

There are several important functions that implement the bulk of the interactive recognition program:

  *_create_energy_images*: Allows user to view energy images in real-time and to save them on demand by pression a number key from 1 to 5.
                         
  *_svm_train*: Reads energy images from a directory and uses them to train the SVM classifier.  Also does post training testing and outputs accuracy results.
              
  *_recognize*: Allows user to view real-time camera input, do hand gestures, and see the classifier output on the screen.
  
No unit tests were created/required for this program as it is interactive with self contained functions.

To run the program in training mode:

    python trainer.py svm_output energy_path
    
    where:
      svm_output = filename of SVM classifier output file
      energy_path = path to energy images

To run the program in recognition mode:

    python trainer.py svm_input
    
    where:
      svm_input = filename of SVM classifier input file  

**NOTE:**

Deprecated code from previous iterations is included in the repository. These include *preprocessor.py*, *recognizer.py*, *tracking_hand.py*, *Tracking_hand_bgsubtraction.py*, and *vidutils.py*.

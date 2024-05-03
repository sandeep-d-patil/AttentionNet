# Training and evaluation on LLAMAS dataset
llamas_scripts include following scripts:

1. llamas_gen.py - This script can be used to predict lanes from llamas images and save the original images with predicted images.
2. llamas_test_ap.py - This script can be used to evaluate the llamas predictions based on metrics suggested by [this repo](https://github.com/cardwing/Codes-for-Lane-Detection).
3. llamas_gen.py - This script is used to 
4. test_index_gen_llamas.py - This script is used to generate the test indices for llamas dataset. 

## Procedure to obtain the average precision, corner precision and corner recall scores
Average precision is average values of precision scores at each threshold value. The average precision is calculated 
as (recall(n) - recall(n-1)) * precision(n). The lane line predictions obtained from the neural network has a 
structure of 256x2x128 which indicates that there are activations from two classes namely foreground and background.

The method to obtain the correct classifications in the evaluation metrics suggested by [qinnzou](https://github.com/qinnzou/Robust-Lane-Detection)
compares the max values in the 2 channels of 256x128 predicted image and return the location of the max value to determine foreground (lane) 
and background. 

The average precision metric requires the output value to be thresholded and determine the foreground (lane) and background.
Hence, max function cannot be used.

`python3 llamas_scripts/llamas_test_ap.py`


`python3 llamas_scripts/test_index_gen_llamas.py`
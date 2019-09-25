# SR_AdaFM_inference
This repository contains a module for inferencing AdaFM model (trained on dataset with profile and frontal faces).

The model: AdaFM.pth
You can download the model <a href="https://drive.google.com/open?id=1N3MIrVrE_svkNrx_RW25VtyypbJqGIcN">here</a> (modified 23.09.2019) or <a href="https://drive.google.com/open?id=10ZYwxAOrzdNP3fKF4iXGAsPLE2Zh8lXS">here</a> (modified 18.09.2019) and move it to SR_AdaFM_inference folder

Requirements: requirements.txt

Inference: codes/inference.py

To inference this model you should import Test class and call inference method with argument path_to_image. You can see the results of working in a folder: "results/".

from inference import Test
test = Test()
test.inference(*path_to_image*)

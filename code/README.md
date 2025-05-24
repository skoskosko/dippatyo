# Scripts

To use these scripts you need to get [cityscapes dataset](https://www.cityscapes-dataset.com/).

Codebase here is quite a mess. I didn't really make it to be edited or used. Only for my own reference.
[cityscapescripts](https://github.com/mcordts/cityscapesScripts) are used by old_scripts. The scripts used in the work itself are in seperate folders.

Lots of these scripts include hard coded paths on my machine. So if you want to use them fix those first. Usually I have hardcodes them in dataset.py files (which yeah, should be a common lib for all of them)

All the requirements should be stored in repo root. so you can just create venv here and install pip stuff, and they should be same for all.

*semantic_segmentation*

Testing for training semantic segmentation model. Basic stuff. nothing special here.

*stereo*

Testing scripts for doing stereo vision. 

*model_generator*

with generate.py and generate_v2.py

You can generate the swipes on the images. It needs depth and semantic data for doing it.

*model_review*

scripts for going trough the generated images. 

*final_model*

Takes the generated selected images and trains the final model from them

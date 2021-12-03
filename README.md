# People-Counting-with-yolov5
> A side project for testing the yolov5 accuracy from [torch hub](https://pytorch.org/hub/ultralytics_yolov5/). 

## Major dependencies
- Python>=3.8
- torch>=1.7
- torchvision
- [motrackers](https://adipandas.github.io/multi-object-tracker/_modules/motrackers/sort_tracker.html) : visit the website to know how to install it.
  - In this project, we use centroid tracker.
- opencv

## Environment setup step:
1. Create a python3.8 environment
2. cd People-Counting-with-yolov5
3. Run `sh setup.sh` to install all the dependencies.
4. Modify the video source path in `configs.yaml` file
5. Run `python main.py`




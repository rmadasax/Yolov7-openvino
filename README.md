# Yolov7-openvino

## Requirements

>	python3 -m venv openvino_env
>	python -m pip install --upgrade pip
>	pip install openvino-dev[pytorch]==2022.3.0
>	pip install openvino-dev[extras]
>	pip install openvino-dev[torch]
>	pip install torch==1.13.1
>	pip install torchvision
>	pip install matplotlib
>	pip install seaborn
>	pip install onnxruntime



## Steps to Execute
### Convert pytorch model in to onnx
>	wget https://github.com/WongKinYiu/yolov7/releases/download/v0.1/yolov7.pt
>	python export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 64
### Convert Onnx model in to IR format using model optimizer
>	mo --input_model yolov7.onnx --reverse_input_channel
sython export.py --weights yolov7.pt --grid --end2end --simplify --topk-all 100 --iou-thres 0.65 --conf-thres 0.35 --img-size 640 640 --max-wh 64

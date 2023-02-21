# Inference for IR model and ONNX model
import argparse     
from openvino.runtime import Core
import cv2
import time
import requests
import random
import numpy as np
import onnxruntime as ort
from PIL import Image
from pathlib import Path
#from collections import OrderedDict,namedtuple
import cv2
names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 
         'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 
         'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 
         'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 
         'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 
         'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 
         'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 
         'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 
         'hair drier', 'toothbrush']
colors = {name:[random.randint(0, 255) for _ in range(3)] for i,name in enumerate(names)}


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    return im, r, (dw, dh)

def preprocess_input(image):
    image, ratio, dwdh = letterbox(image, auto=False)
    image = image.transpose((2, 0, 1))
    image = np.expand_dims(image, 0)
    image = np.ascontiguousarray(image)
    im = image.astype(np.float32)
    im /= 255
    im.shape
    return im,ratio,dwdh

def infer_onnxmodel(weight,OV_backend,image):    
    w = weight
    providers = ['OpenVINOExecutionProvider', 'CPUExecutionProvider'] if OV_backend else ['CPUExecutionProvider']
    session = ort.InferenceSession(w, providers=providers)
   
    outname = [i.name for i in session.get_outputs()]
    #outname

    inname = [i.name for i in session.get_inputs()]
    #inname
    inp = {inname[0]:image}

    outputs = session.run(outname, inp)[0]
    #outputs
    return outputs

def infer_IRmodel(weight,OV_backend,image):    

    ie = Core()
    #yolov7classification_model_xml = "yolov7/yolov7.xml"
    #model = ie.read_model(model=classification_model_xml)
    model = ie.read_model(model=weight)
    output_layer = model.output(0)
    input_layer = model.input(0)
    compiled_model = ie.compile_model(model=model, device_name="CPU")
    request = compiled_model.create_infer_request()
    request.infer(inputs={input_layer.any_name: im})
    result = request.get_output_tensor(output_layer.index).data
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', required=True, help='model path')
    parser.add_argument('--type', required=True, help='model Type: ONNX | IR')
    parser.add_argument('--input', required=True, help='Input file Path')
    parser.add_argument('--backend', default='ov', help='backend ov | cpu')
    parser.add_argument('--dtype', default='FP32', help='data type : FP32 | FP16')

    args = parser.parse_args()
    print(args.input)
    img = cv2.imread(args.input)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    image = img.copy()
    OV=True 
    if args.backend == 'cpu' : 
       OV=False 
    


    #Pre Process the frames for Openvino# 
    im,ratio,dwdh=preprocess_input(image)
    

    #Inference using IR or ONNX 
    if args.type == 'IR' : 
      result=infer_IRmodel(args.model,OV,im)    
    else : 
      result=infer_onnxmodel(args.model,OV,im)    
   
    #Process the Output 
    ori_images = [img.copy()]

    for i,(batch_id,x0,y0,x1,y1,cls_id,score) in enumerate(result):
        image = ori_images[int(batch_id)]
        box = np.array([x0,y0,x1,y1])
        box -= np.array(dwdh*2)
        box /= ratio
        box = box.round().astype(np.int32).tolist()
        cls_id = int(cls_id)
        score = round(float(score),3)
        name = names[cls_id]
        color = colors[name]
        name += ' '+str(score)
        cv2.rectangle(image,box[:2],box[2:],color,2)
        cv2.putText(image,name,(box[0], box[1] - 2),cv2.FONT_HERSHEY_SIMPLEX,0.75,[225, 255, 255],thickness=2)  
    #Save the output
    im=Image.fromarray(ori_images[0])
    im.save("your_file.jpeg")
    print(args.type)

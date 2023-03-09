from openvino.runtime import Core
import re
import cv2
import numpy as np
import random
import yolov7
from utils.general import scale_coords, non_max_suppression
from utils.metrics import ap_per_class
import argparse
import torch
import time
import yaml
from openvino.runtime import Model
from tqdm import tqdm
from collections import namedtuple
from utils.general import check_dataset, box_iou, xywh2xyxy, colorstr
from utils.datasets import create_dataloader
from openvino.preprocess import PrePostProcessor, ColorFormat
from openvino.runtime import Layout, AsyncInferQueue, PartialShape
# read dataset config
DATA_CONFIG = 'data/coco.yaml'
with open(DATA_CONFIG) as f:
    testdata = yaml.load(f, Loader=yaml.SafeLoader)
# Dataloader
TASK = 'val'  # path to train/val/test images
Option = namedtuple('Options', ['single_cls'])  # imitation of commandline provided options for single class evaluation
opt = Option(False)
dataloader = create_dataloader(
    testdata[TASK], 640, 1, 32, opt, pad=0.5,
    prefix=colorstr(f'{TASK}: ')
)[0]

class YOLOV7_OPENVINO(object):
    def __init__(self, model_path, device, pre_api, batchsize, nireq, grid):
        # set the hyperparameters
        self.classes = [
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
       ]
        self.batchsize = batchsize
        self.grid = grid
        self.img_size = (640, 640) 
        self.conf_thres = 0.001
        self.iou_thres = 0.65
        self.class_num = 80
        self.colors = [[random.randint(0, 255) for _ in range(3)] for _ in self.classes]
        self.stride = [8, 16, 32]
        self.anchor_list = [[12, 16, 19, 36, 40, 28], [36, 75, 76, 55, 72, 146], [142, 110, 192, 243, 459, 401]]
        self.anchor = np.array(self.anchor_list).astype(float).reshape(3, -1, 2)
        area = self.img_size[0] * self.img_size[1]
        self.size = [int(area / self.stride[0] ** 2), int(area / self.stride[1] ** 2), int(area / self.stride[2] ** 2)]
        self.feature = [[int(j / self.stride[i]) for j in self.img_size] for i in range(3)]

        ie = Core()
        self.model = ie.read_model(model_path)
        self.input_layer = self.model.input(0)
        new_shape = PartialShape([self.batchsize, 3, self.img_size[0], self.img_size[1]])
        self.model.reshape({self.input_layer.any_name: new_shape})
        self.pre_api = pre_api
        if (self.pre_api == True):
            # Preprocessing API
            ppp = PrePostProcessor(self.model)
            # Declare section of desired application's input format
            ppp.input().tensor() \
                .set_layout(Layout("NHWC")) \
                .set_color_format(ColorFormat.BGR)
            # Here, it is assumed that the model has "NCHW" layout for input.
            ppp.input().model().set_layout(Layout("NCHW"))
            # Convert current color format (BGR) to RGB
            ppp.input().preprocess() \
                .convert_color(ColorFormat.RGB) \
                .scale([255.0, 255.0, 255.0])
            self.model = ppp.build()
            print(f'Dump preprocessor: {ppp}')

        self.compiled_model = ie.compile_model(model=self.model, device_name=device)
        self.infer_queue = AsyncInferQueue(self.compiled_model, nireq)
        self.request = self.compiled_model.create_infer_request()
    def letterbox(self, img, new_shape=(640, 640), color=(114, 114, 114)):
        # Resize and pad image while meeting stride-multiple constraints
        shape = img.shape[:2]  # current shape [height, width]
        if isinstance(new_shape, int):
            new_shape = (new_shape, new_shape)

        # Scale ratio (new / old)
        r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
        new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
        dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - \
            new_unpad[1]  # wh padding

        # divide padding into 2 sides
        dw /= 2
        dh /= 2

        # resize
        if shape[::-1] != new_unpad:
            img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
        top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
        left, right = int(round(dw - 0.1)), int(round(dw + 0.1))

        # add border
        img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)

        return img

    def xywh2xyxy(self, x):
        # Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y

        return y

    def nms(self, prediction, conf_thres, iou_thres):
        predictions = np.squeeze(prediction[0])

        # Filter out object confidence scores below threshold
        obj_conf = predictions[:, 4]
        predictions = predictions[obj_conf > conf_thres]
        obj_conf = obj_conf[obj_conf > conf_thres]

        # Multiply class confidence with bounding box confidence
        predictions[:, 5:] *= obj_conf[:, np.newaxis]

        # Get the scores
        scores = np.max(predictions[:, 5:], axis=1)

        # Filter out the objects with a low score
        valid_scores = scores > conf_thres
        predictions = predictions[valid_scores]
        scores = scores[valid_scores]

        # Get the class with the highest confidence
        class_ids = np.argmax(predictions[:, 5:], axis=1)

        # Get bounding boxes for each object
        boxes = self.xywh2xyxy(predictions[:, :4])

        # Apply non-maxima suppression to suppress weak, overlapping bounding boxes
        # indices = nms(boxes, scores, self.iou_threshold)
        indices = cv2.dnn.NMSBoxes(boxes.tolist(), scores.tolist(), conf_thres, iou_thres)

        return boxes[indices], scores[indices], class_ids[indices]

    def clip_coords(self, boxes, img_shape):
        # Clip bounding xyxy bounding boxes to image shape (height, width)
        boxes[:, 0].clip(0, img_shape[1])  # x1
        boxes[:, 1].clip(0, img_shape[0])  # y1
        boxes[:, 2].clip(0, img_shape[1])  # x2
        boxes[:, 3].clip(0, img_shape[0])  # y2

    def scale_coords(self, img1_shape, img0_shape, coords, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
        # gain  = old / new
        if ratio_pad is None:
            gain = min(img1_shape[0] / img0_shape[0],
                       img1_shape[1] / img0_shape[1])
            padding = (img1_shape[1] - img0_shape[1] * gain) / \
                2, (img1_shape[0] - img0_shape[0] * gain) / 2
        else:
            gain = ratio_pad[0][0]
            padding = ratio_pad[1]
        coords[:, [0, 2]] -= padding[0]  # x padding
        coords[:, [1, 3]] -= padding[1]  # y padding
        coords[:, :4] /= gain
        self.clip_coords(coords, img0_shape)

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def plot_one_box(self, x, img, color=None, label=None, line_thickness=None):
        # Plots one bounding box on image img
        tl = line_thickness or round(
            0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
        color = color or [random.randint(0, 255) for _ in range(3)]
        c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
        cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
        if label:
            tf = max(tl - 1, 1)  # font thickness
            t_size = cv2.getTextSize(
                label, 0, fontScale=tl / 3, thickness=tf)[0]
            c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
            cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3,
                        [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)
    
    def draw1(self, img, boxinfo):
        for xyxy, conf, cls in boxinfo:
            self.plot_one_box(xyxy, img, label=self.classes[int(cls)], color=self.colors[int(cls)], line_thickness=2)
        cv2.imwrite("yolov7_out_test.jpg", img)
        #cv2.imshow('Press ESC to Exit', img) 
        #cv2.waitKey(1)

    def postprocess_stats(self, infer_request, info):
        src_img_list, src_size = info
        for batch_id in range(self.batchsize):
            if self.grid:
                results = np.expand_dims(infer_request.get_output_tensor(0).data[batch_id], axis=0)
            else:
                output = []
                # Get the each feature map's output data
                output.append(self.sigmoid(self.request.get_output_tensor(0).data[batch_id].reshape(-1, self.size[0]*3, 5+self.class_num)))
                output.append(self.sigmoid(self.request.get_output_tensor(1).data[batch_id].reshape(-1, self.size[1]*3, 5+self.class_num)))
                output.append(self.sigmoid(self.request.get_output_tensor(2).data[batch_id].reshape(-1, self.size[2]*3, 5+self.class_num)))
                # Postprocessing
                grid = []
                for _, f in enumerate(self.feature):
                    grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

                result = []
                for i in range(3):
                    src = output[i]
                    xy = src[..., 0:2] * 2. - 0.5
                    wh = (src[..., 2:4] * 2) ** 2
                    dst_xy = []
                    dst_wh = []
                    for j in range(3):
                        dst_xy.append((xy[:, j * self.size[i]:(j + 1) * self.size[i], :] + grid[i]) * self.stride[i])
                        dst_wh.append(wh[:, j * self.size[i]:(j + 1) *self.size[i], :] * self.anchor[i][j])
                    src[..., 0:2] = np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1)
                    src[..., 2:4] = np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1)
                    result.append(src)
                results = np.concatenate(result, 1)
                resulto=torch.from_numpy(results)
            outs = non_max_suppression(resulto, conf_thres=self.conf_thres, iou_thres=self.iou_thres, labels=None, multi_label=True)
            #boxes, scores, class_ids = self.nms(results, self.conf_thres, self.iou_thres)
            img_shape = self.img_size
            #self.scale_coords(img_shape, src_size, boxes)
            
            # Draw the results
            #self.draw(src_img_list[batch_id], zip(boxes, scores, class_ids))
            return outs  



    def infer_image_stats(self, img_path):
        # Read image
        out=[]
        start_time = time.time()
        src_img = cv2.imread(img_path)
        #src_img = img_path
        src_img_list = []
        src_img_list.append(src_img)
        img = src_img
        src_size = src_img.shape[:2]
        img = self.letterbox(src_img, self.img_size)
        if(self.pre_api == False):
           img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
           img = img.transpose(2, 0, 1) # NHWC to NCHW
           img = img.astype(dtype=np.float32)
           img /= 255.0
        input_image = np.expand_dims(img, 0)
        self.request.infer(inputs={self.input_layer.any_name: input_image})
        results0 = self.request.get_output_tensor(0).data
        results1 = self.request.get_output_tensor(1).data
        infer_request=self.request
        self.infer_queue.wait_all()
        #out=self.postprocess_stats(out,(src_img_list, src_size)) 
        out=self.postprocess_stats(infer_request,(src_img_list, src_size)) 
        return out 
    def postprocess(self, infer_request, info):
        src_img_list, src_size = info
        for batch_id in range(self.batchsize):
            if self.grid:
                results = np.expand_dims(infer_request.get_output_tensor(0).data[batch_id], axis=0)
            else:
                output = []
                # Get the each feature map's output data
                output.append(self.sigmoid(infer_request.get_output_tensor(0).data[batch_id].reshape(-1, self.size[0]*3, 5+self.class_num)))
                output.append(self.sigmoid(infer_request.get_output_tensor(1).data[batch_id].reshape(-1, self.size[1]*3, 5+self.class_num)))
                output.append(self.sigmoid(infer_request.get_output_tensor(2).data[batch_id].reshape(-1, self.size[2]*3, 5+self.class_num)))
                # Postprocessing
                grid = []
                for _, f in enumerate(self.feature):
                    grid.append([[i, j] for j in range(f[0]) for i in range(f[1])])

                result = []
                for i in range(3):
                    src = output[i]
                    xy = src[..., 0:2] * 2. - 0.5
                    wh = (src[..., 2:4] * 2) ** 2
                    dst_xy = []
                    dst_wh = []
                    for j in range(3):
                        dst_xy.append((xy[:, j * self.size[i]:(j + 1) * self.size[i], :] + grid[i]) * self.stride[i])
                        dst_wh.append(wh[:, j * self.size[i]:(j + 1) *self.size[i], :] * self.anchor[i][j])
                    src[..., 0:2] = np.concatenate((dst_xy[0], dst_xy[1], dst_xy[2]), axis=1)
                    src[..., 2:4] = np.concatenate((dst_wh[0], dst_wh[1], dst_wh[2]), axis=1)
                    result.append(src)
                results = np.concatenate(result, 1)
            boxes, scores, class_ids = self.nms(results, self.conf_thres, self.iou_thres)
            img_shape = self.img_size
            self.scale_coords(img_shape, src_size, boxes)
            # Draw the results
            #self.draw(src_img_list[batch_id], zip(boxes, scores, class_ids))

    def infer_image(self, img_path):
        # Read image
        start_time = time.time()
        src_img = cv2.imread(img_path)
        src_img_list = []
        src_img_list.append(src_img)
        img = self.letterbox(src_img, self.img_size)
        src_size = src_img.shape[:2]
        img = img.astype(dtype=np.float32)
        if (self.pre_api == False):
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # BGR to RGB
            img /= 255.0
            img = img.transpose(2, 0, 1) # NHWC to NCHW
        input_image = np.expand_dims(img, 0)

        # Set callback function for postprocess
        self.infer_queue.set_callback(self.postprocess)
        # Do inference
        self.infer_queue.start_async({self.input_layer.any_name: input_image}, (src_img_list, src_size))
        self.infer_queue.wait_all()
        end_time = time.time()
        # Calculate the average FPS\n",
        fps = 1 / (end_time - start_time)
        print("throughput: {:.2f} fps".format(fps))
        cv2.imwrite("yolov7_out.jpg", src_img_list[0])


def test(data,
         YOLOV7,
        model=Model,
         dataloader= torch.utils.data.DataLoader,
         conf_thres = 0.001,
         iou_thres= 0.65,  # for NMS
         single_cls = False,
         v5_metric = False,
         names= None
        ):
    """
    YOLOv7 accuracy evaluation. Processes validation dataset and compites metrics.
    
    Parameters:
        model (Model): OpenVINO compiled model.
        dataloader (torch.utils.DataLoader): validation dataset.
        conf_thres (float, *optional*, 0.001): minimal confidence threshold for keeping detections
        iou_thres (float, *optional*, 0.65): IOU threshold for NMS
        single_cls (bool, *optional*, False): class agnostic evaluation
        v5_metric (bool, *optional*, False): use YOLOv5 evaluation approach for metrics calculation
        names (List[str], *optional*, None): names for each class in dataset
    Returns:
        mp (float): mean precision
        mr (float): mean recall
        map50 (float): mean average precision at 0.5 IOU threshold
        map (float): mean average precision at 0.5:0.95 IOU thresholds
        maps (Dict(int, float): average precision per class
        seen (int): number of evaluated images
        labels (int): number of labels
    """

    check_dataset(data)  # check
    nc = 1 if single_cls else int(data['nc'])  # number of classes
    iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
    niou = iouv.numel()

    if v5_metric:
        print("Testing with YOLOv5 AP metric...")
    
    seen = 0
    p, r, mp, mr, map50, map = 0., 0., 0., 0., 0., 0.
    stats, ap, ap_class = [], [], []
    for (img, targets, path, shapes) in tqdm(dataloader):
        paths=''.join(path)
        print(paths)
        out=YOLOV7.infer_image_stats(paths)
        targets = targets
        height, width = img.shape[2:]

        with torch.no_grad():
            # Run model
            #out = torch.from_numpy(model(img)[model_output])  # inference output            
            # Run NMS
            targets[:, 2:] *= torch.Tensor([width, height, width, height])  # to pixels

            #out = non_max_suppression(out, conf_thres=conf_thres, iou_thres=iou_thres, labels=None, multi_label=True)
        for si, pred in enumerate(out):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if len(pred) == 0:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue
            # Predictions
            predn = pred.clone()
            scale_coords(img[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred
            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device='cpu')
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]
                tbox = xywh2xyxy(labels[:, 1:5])
                scale_coords(img[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels
                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # target indices
                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices
                        # Append detections
                        detected_set = set()
                        for j in (ious > iouv[0]).nonzero(as_tuple=False):
                            d = ti[i[j]]  # detected target
                            if d.item() not in detected_set:
                                detected_set.add(d.item())
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break
            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
            #stats.append((correct.cpu(), boxes, class_ids[si], tcls))
    # Compute statistics
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats) and stats[0].any():
        p, r, ap, f1, ap_class = ap_per_class(*stats, plot=True, v5_metric=v5_metric, names=names)
        ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
        mp, mr, map50, map = p.mean(), r.mean(), ap50.mean(), ap.mean()
        print( mp, mr, map50, map)
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return mp, mr, map50, map, maps, seen, nt.sum()
    

if __name__ == "__main__":
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('-h', '--help', action='help', help='Show this help message and exit.')
    parser.add_argument('-i', '--input', required=True, type=str,
                      help='Required. Path to an image file.')
    parser.add_argument('-m', '--model', required=True, type=str,
                      help='Required. Path to an .xml or .onnx file with a trained model.')
    parser.add_argument('-d', '--device', required=False, default='CPU', type=str,
                      help='Device name.')
    parser.add_argument('-p', '--pre_api', required=False, action='store_true', 
                      help='Preprocessing api.')
    parser.add_argument('-g', '--grid', required=False, action='store_true', 
                      help='With grid in model.')
    args = parser.parse_args()
    print(args.pre_api,args.grid)
    yolov7_detector=YOLOV7_OPENVINO(args.model, args.device, args.pre_api, 1, 1, args.grid)
    #Enable this to measure the FPS
    #yolov7_detector.infer_image(args.input)
    #Enable this to measure the Accuracy
    FP32_result = test(data=testdata, YOLOV7=yolov7_detector, model=args.model, dataloader=dataloader, names=yolov7_detector.classes)
    mp, mr, map50, map, maps, num_images, labels = FP32_result
    s = ('%20s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'Precision', 'Recall', 'mAP@.5', 'mAP@.5:.95')
    print(s)
    pf = '%20s' + '%12i' * 2 + '%12.3g' * 4  # print format
    print(pf % ('all', num_images, labels, mp, mr, map50, map))


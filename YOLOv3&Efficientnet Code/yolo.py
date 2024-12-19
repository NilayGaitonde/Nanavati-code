import colorsys
import os
import time

import numpy as np
import torch
import torch.nn as nn
from PIL import ImageDraw, ImageFont

from nets.yolo import YoloBody
from utils.utils import (cvtColor, get_anchors, get_classes, preprocess_input,
                         resize_image, show_config)
from utils.utils_bbox import DecodeBox

'''
Important note for training your own dataset!
'''
class YOLO(object):
    _defaults = {
        #--------------------------------------------------------------------------#
        #   When using your own trained model for prediction, you must modify model_path and classes_path!
        #   model_path points to the weights file in the logs folder, classes_path points to the txt in model_data
        #
        #   After training, there are multiple weight files in the logs folder, choose the one with lower validation set loss.
        #   Lower validation set loss doesn't mean higher mAP, it only means better generalization on the validation set.
        #   If shape mismatch occurs, also pay attention to modifying model_path and classes_path parameters during training
        #--------------------------------------------------------------------------#
        "model_path"        : '/Users/nilaygaitonde/Documents/Projects/Nanavati/Data/new/YOLOv3&Efficientnet Code/logs/best_epoch_weights.pth',
        "classes_path"      : '/Users/nilaygaitonde/Documents/Projects/Nanavati/Data/new/YOLOv3&Efficientnet Code/model_data/voc_classes.txt',
        #---------------------------------------------------------------------#
        #   anchors_path represents the txt file corresponding to prior boxes, generally no modification needed.
        #   anchors_mask helps code find corresponding prior boxes, generally no modification needed.
        #---------------------------------------------------------------------#
        "anchors_path"      : '/Users/nilaygaitonde/Documents/Projects/Nanavati/Data/new/YOLOv3&Efficientnet Code/model_data/yolo_anchors.txt',
        "anchors_mask"      : [[6, 7, 8], [3, 4, 5], [0, 1, 2]],
        #---------------------------------------------------------------------#
        #   Input image size must be a multiple of 32.
        #---------------------------------------------------------------------#
        "input_shape"       : [416, 416],
        #---------------------------------------------------------------------#
        #   EfficientNet version
        #   phi = 0 represents efficientnet-B0-yolov3
        #   phi = 1 represents efficientnet-B1-yolov3
        #   phi = 2 represents efficientnet-B2-yolov3   
        #   ... and so on
        #---------------------------------------------------------------------#
        "phi"               : 2,
        #---------------------------------------------------------------------#
        #   Only prediction boxes with scores higher than confidence will be kept
        #---------------------------------------------------------------------#
        "confidence"        : 0.5,
        #---------------------------------------------------------------------#
        #   NMS IOU size used for non-maximum suppression
        #---------------------------------------------------------------------#
        "nms_iou"           : 0.3,
        #---------------------------------------------------------------------#
        #   This variable controls whether to use letterbox_image for non-distorted resizing of input images,
        #   After multiple tests, turning off letterbox_image and direct resizing shows better results
        #---------------------------------------------------------------------#
        "letterbox_image"   : False,
        #-------------------------------#
        #   Whether to use CUDA
        #   Set to False if no GPU available
        #-------------------------------#
        "cuda"              : False,
    }

    @classmethod
    def get_defaults(cls, n):
        if n in cls._defaults:
            return cls._defaults[n]
        else:
            return "Unrecognized attribute name '" + n + "'"

    #---------------------------------------------------#
    #   Initialize YOLO
    #---------------------------------------------------#
    def __init__(self, **kwargs):
        self.__dict__.update(self._defaults)
        for name, value in kwargs.items():
            setattr(self, name, value)
            
        #---------------------------------------------------#
        #   Get number of classes and prior boxes
        #---------------------------------------------------#
        self.class_names, self.num_classes  = get_classes(self.classes_path)
        self.anchors, self.num_anchors      = get_anchors(self.anchors_path)
        self.bbox_util                      = DecodeBox(self.anchors, self.num_classes, (self.input_shape[0], self.input_shape[1]), self.anchors_mask)

        #---------------------------------------------------#
        #   Set different colors for drawing boxes
        #---------------------------------------------------#
        hsv_tuples = [(x / self.num_classes, 1., 1.) for x in range(self.num_classes)]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)), self.colors))
        self.generate()
        
        show_config(**self._defaults)

    #---------------------------------------------------#
    #   Generate model
    #---------------------------------------------------#
    def generate(self):
        #---------------------------------------------------#
        #   Build YOLOv3 model and load weights
        #---------------------------------------------------#
        self.net    = YoloBody(self.anchors_mask, self.num_classes, phi = self.phi)
        device      = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.net.load_state_dict(torch.load(self.model_path, map_location=device))
        self.net    = self.net.eval()
        print('{} model, anchors, and classes loaded.'.format(self.model_path))
        if self.cuda:
            self.net = nn.DataParallel(self.net)
            self.net = self.net.cuda()

    #---------------------------------------------------#
    #   Detect image
    #---------------------------------------------------#
    def detect_image(self, image, crop = False, count = False):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Convert image to RGB here to prevent grayscale images from causing errors during prediction.
        #   Code only supports RGB image prediction, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to achieve non-distorted resize
        #   Can also directly resize for detection
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Input image into network for prediction!
            #---------------------------------------------------------#
            outputs = self.net(images)
            # outputs will have a three element tuple (out1 for large objects, out2 for medium objects, out3 for small objects)
            # each output has a shape of (batch_size, num_anchors * (num_classes + 5), feature_map_size) -> (1,507,6)
            outputs = self.bbox_util.decode_box(outputs)
            # max_bleh = outputs[0]
            # print("Confidence:",max_bleh[:,4][0][torch.argmax(max_bleh[:,4]).item()])
            # medium_bleh = outputs[1]
            # print("Confidence:",medium_bleh[:,4][0][torch.argmax(medium_bleh[:,4])])
            # min_bleh = outputs[2]
            # print("Confidence:",min_bleh[:,4][0][torch.argmax(min_bleh[:,4])])
            #---------------------------------------------------------#
            #   Stack prediction boxes then perform non-maximum suppression
            #---------------------------------------------------------#
            # results = outputs
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
            
            # print(results)
            if results[0] is None: 
                return image

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]
        #---------------------------------------------------------#
        #   Set font and border thickness
        #---------------------------------------------------------#
        font        = ImageFont.truetype(font='Data/new/YOLOv3&Efficientnet Code/model_data/simhei.ttf', size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness   = int(max((image.size[0] + image.size[1]) // np.mean(self.input_shape), 1))
        #---------------------------------------------------------#
        #   Count
        #---------------------------------------------------------#
        if count:
            print("top_label:", top_label)
            classes_nums    = np.zeros([self.num_classes])
            for i in range(self.num_classes):
                num = np.sum(top_label == i)
                if num > 0:
                    print(self.class_names[i], " : ", num)
                classes_nums[i] = num
            print("classes_nums:", classes_nums)
        #---------------------------------------------------------#
        #   Whether to crop objects
        #---------------------------------------------------------#
        if crop:
            for i, c in list(enumerate(top_label)):
                top, left, bottom, right = top_boxes[i]
                top     = max(0, np.floor(top).astype('int32'))
                left    = max(0, np.floor(left).astype('int32'))
                bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
                right   = min(image.size[0], np.floor(right).astype('int32'))
                
                dir_save_path = "img_crop"
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                crop_image = image.crop([left, top, right, bottom])
                crop_image.save(os.path.join(dir_save_path, "crop_" + str(i) + ".png"), quality=95, subsampling=0)
                print("save crop_" + str(i) + ".png to " + dir_save_path)
        #---------------------------------------------------------#
        #   Draw image
        #---------------------------------------------------------#
        print(top_label)
        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = top_conf[i]

            top, left, bottom, right = box

            top     = max(0, np.floor(top).astype('int32'))
            left    = max(0, np.floor(left).astype('int32'))
            bottom  = min(image.size[1], np.floor(bottom).astype('int32'))
            right   = min(image.size[0], np.floor(right).astype('int32'))

            label = '{} {:.2f}'.format(predicted_class, score)
            print(label)
            draw = ImageDraw.Draw(image)
            label_size = draw.textlength(label, font)
            label = label.encode('utf-8')
            print(label, top, left, bottom, right)
            
            # if top - label_size[1] >= 0:
            #     text_origin = np.array([left, top - label_size[1]])
            if top - label_size >= 0:
                text_origin = np.array([left, top - label_size])
            else:
                text_origin = np.array([left, top + 1])

            for i in range(thickness):
                draw.rectangle([left + i, top + i, right - i, bottom - i], outline=self.colors[c])
            draw.rectangle([tuple(text_origin), tuple(text_origin + label_size)], fill=self.colors[c])
            draw.text(text_origin, str(label,'UTF-8'), fill=(0, 0, 0), font=font)
            del draw

        return image

    def get_FPS(self, image, test_interval):
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Convert image to RGB here to prevent grayscale images from causing errors during prediction.
        #   Code only supports RGB image prediction, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to achieve non-distorted resize
        #   Can also directly resize for detection
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Input image into network for prediction!
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   Stack prediction boxes then perform non-maximum suppression
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                                                    
        t1 = time.time()
        for _ in range(test_interval):
            with torch.no_grad():
                #---------------------------------------------------------#
                #   Input image into network for prediction!
                #---------------------------------------------------------#
                outputs = self.net(images)
                outputs = self.bbox_util.decode_box(outputs)
                #---------------------------------------------------------#
                #   Stack prediction boxes then perform non-maximum suppression
                #---------------------------------------------------------#
                results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                            image_shape, self.letterbox_image, conf_thres=self.confidence, nms_thres=self.nms_iou)
                            
        t2 = time.time()
        tact_time = (t2 - t1) / test_interval
        return tact_time

    def detect_heatmap(self, image, heatmap_save_path):
        import cv2
        import matplotlib.pyplot as plt
        def sigmoid(x):
            y = 1.0 / (1.0 + np.exp(-x))
            return y
        #---------------------------------------------------------#
        #   Convert image to RGB here to prevent grayscale images from causing errors during prediction.
        #   Code only supports RGB image prediction, all other types will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to achieve non-distorted resize
        #   Can also directly resize for detection
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Feed the image into the network and make predictions!
            #---------------------------------------------------------#
            outputs = self.net(images)
        
        plt.imshow(image, alpha=1)
        plt.axis('off')
        mask    = np.zeros((image.size[1], image.size[0]))
        for sub_output in outputs:
            sub_output = sub_output.cpu().numpy()
            b, c, h, w = np.shape(sub_output)
            sub_output = np.transpose(np.reshape(sub_output, [b, 3, -1, h, w]), [0, 3, 4, 1, 2])[0]
            score      = np.max(sigmoid(sub_output[..., 4]), -1)
            score      = cv2.resize(score, (image.size[0], image.size[1]))
            normed_score    = (score * 255).astype('uint8')
            mask            = np.maximum(mask, normed_score)
            
        plt.imshow(mask, alpha=0.5, interpolation='nearest', cmap="jet")

        plt.axis('off')
        plt.subplots_adjust(top=1, bottom=0, right=1,  left=0, hspace=0, wspace=0)
        plt.margins(0, 0)
        plt.savefig(heatmap_save_path, dpi=200, bbox_inches='tight', pad_inches = -0.1)
        print("Save to the " + heatmap_save_path)
        plt.show()

    def get_map_txt(self, image_id, image, class_names, map_out_path):
        f = open(os.path.join(map_out_path, "detection-results/"+image_id+".txt"),"w") 
        image_shape = np.array(np.shape(image)[0:2])
        #---------------------------------------------------------#
        #   Here, the image is converted to RGB image to prevent grayscale image from reporting errors during prediction.
        #   The code only supports prediction of RGB images, and all other types of images will be converted to RGB
        #---------------------------------------------------------#
        image       = cvtColor(image)
        #---------------------------------------------------------#
        #   Add gray bars to the image to achieve distortion-free resizing
        #   You can also resize directly for recognition
        #---------------------------------------------------------#
        image_data  = resize_image(image, (self.input_shape[1],self.input_shape[0]), self.letterbox_image)
        #---------------------------------------------------------#
        #   Add the batch_size dimension
        #---------------------------------------------------------#
        image_data  = np.expand_dims(np.transpose(preprocess_input(np.array(image_data, dtype='float32')), (2, 0, 1)), 0)

        with torch.no_grad():
            images = torch.from_numpy(image_data)
            if self.cuda:
                images = images.cuda()
            #---------------------------------------------------------#
            #   Feed the image into the network and make predictions!
            #---------------------------------------------------------#
            outputs = self.net(images)
            outputs = self.bbox_util.decode_box(outputs)
            #---------------------------------------------------------#
            #   The prediction boxes are stacked and then non-maximum suppression is performed
            #---------------------------------------------------------#
            results = self.bbox_util.non_max_suppression(torch.cat(outputs, 1), self.num_classes, self.input_shape, 
                        image_shape, self.letterbox_image, conf_thres = self.confidence, nms_thres = self.nms_iou)
                                                    
            if results[0] is None: 
                return 

            top_label   = np.array(results[0][:, 6], dtype = 'int32')
            top_conf    = results[0][:, 4] * results[0][:, 5]
            top_boxes   = results[0][:, :4]

        for i, c in list(enumerate(top_label)):
            predicted_class = self.class_names[int(c)]
            box             = top_boxes[i]
            score           = str(top_conf[i])

            top, left, bottom, right = box
            if predicted_class not in class_names:
                continue

            f.write("%s %s %s %s %s %s\n" % (predicted_class, score[:6], str(int(left)), str(int(top)), str(int(right)),str(int(bottom))))

        f.close()
        return 

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data.dataset import Dataset

from utils import cvtColor, preprocess_input


class YoloDataset(Dataset):
    def __init__(self, annotation_lines, input_shape, num_classes, train):
        super(YoloDataset, self).__init__()
        self.annotation_lines   = annotation_lines
        self.input_shape        = input_shape
        self.num_classes        = num_classes
        self.length             = len(self.annotation_lines)
        self.train              = train

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        index       = index % self.length
        #---------------------------------------------------#
        #   训练时进行数据的随机增强
        #   验证时不进行数据的随机增强
        #---------------------------------------------------#
        image, box  = self.get_random_data(self.annotation_lines[index], self.input_shape[0:2], random = self.train)
        image       = np.transpose(preprocess_input(np.array(image, dtype=np.float32)), (2, 0, 1))
        box         = np.array(box, dtype=np.float32)
        if len(box) != 0:
            box[:, [0, 2]] = box[:, [0, 2]] / self.input_shape[1]
            box[:, [1, 3]] = box[:, [1, 3]] / self.input_shape[0]

            box[:, 2:4] = box[:, 2:4] - box[:, 0:2]
            box[:, 0:2] = box[:, 0:2] + box[:, 2:4] / 2
        return image, box

    def rand(self, a=0, b=1):
        return np.random.rand()*(b-a) + a

    def get_random_data(self, annotation_line, input_shape, jitter=.3, hue=.1, sat=0.7, val=0.4, random=True):
        line    = annotation_line.split()
        #------------------------------#
        #   读取图像并转换成RGB图像
        #------------------------------#
        image   = Image.open(line[0])
        image   = cvtColor(image)
        #------------------------------#
        #   获得图像的高宽与目标高宽
        #------------------------------#
        iw, ih  = image.size
        h, w    = input_shape
        #------------------------------#
        #   获得预测框
        #------------------------------#
        box     = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

        if not random:
            scale = min(w/iw, h/ih) # scale = 416/1024 (0.4) or 416/768(0.5) -> 0.4
            nw = int(iw*scale) # 1024*0.4 = 410
            nh = int(ih*scale) # 768*0.4 = 307
            dx = (w-nw)//2 # (416-410)//2 = 3
            dy = (h-nh)//2 # (416-307)//2 = 54

            #---------------------------------#
            #   将图像多余的部分加上灰条
            #---------------------------------#
            image       = image.resize((nw,nh), Image.BICUBIC) # 410, 307
            new_image   = Image.new('RGB', (w,h), (128,128,128)) # 416, 416
            new_image.paste(image, (dx, dy)) # resized image pasted on new_image (which is a basic gray image) 3 from left, 54 from top
            image_data  = np.array(new_image, np.float32) # 416, 416, 3

            #---------------------------------#
            #   对真实框进行调整
            #---------------------------------#
            if len(box)>0: # if there are boxes
                # adjust scaling of boxes
                np.random.shuffle(box) # shuffle the boxes box = [467,371,515,427]
                box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx # box[:,[0,2]] = [467,515]*410/1024 + 3 = [187, 515]
                box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy # box[:,[1,3]] = [371,427]*307/768 + 54 = [187, 427]
                box[:, 0:2][box[:, 0:2]<0] = 0 # box[:, 0:2] = [187, 187] if box[:, 0:2] < 0 then box[:, 0:2] = 0
                box[:, 2][box[:, 2]>w] = w # box[:, 2] = 515 if box[:, 2] > 416 then box[:, 2] = 416
                box[:, 3][box[:, 3]>h] = h # box[:, 3] = 427 if box[:, 3] > 416 then box[:, 3] = 416
                box_w = box[:, 2] - box[:, 0] # box_w = 515 - 187 = 328
                box_h = box[:, 3] - box[:, 1] # box_h = 427 - 187 = 240
                box = box[np.logical_and(box_w>1, box_h>1)] # discard invalid box 

            return image_data, box
                
        #------------------------------------------#
        #   对图像进行缩放并且进行长和宽的扭曲
        #------------------------------------------#
        new_ar = iw/ih * self.rand(1-jitter,1+jitter) / self.rand(1-jitter,1+jitter) # new_ar = 1024/768 * 0.7 / 0.7 = 1
        scale = self.rand(.25, 2) # scale = 0.25 + 1.75 = 2
        if new_ar < 1: #if height is greater than width then we need to adjust more height
            nh = int(scale*h) 
            nw = int(nh*new_ar)
        else:
            nw = int(scale*w)
            nh = int(nw/new_ar)
        image = image.resize((nw,nh), Image.BICUBIC) # resize the image to new width and height

        #------------------------------------------#
        #   将图像多余的部分加上灰条 add grey padding to the image
        #------------------------------------------#
        dx = int(self.rand(0, w-nw))
        dy = int(self.rand(0, h-nh))
        new_image = Image.new('RGB', (w,h), (128,128,128))
        new_image.paste(image, (dx, dy))
        image = new_image

        #------------------------------------------#
        #   翻转图像
        #------------------------------------------#
        flip = self.rand()<.5
        if flip: image = image.transpose(Image.FLIP_LEFT_RIGHT)

        image_data      = np.array(image, np.uint8)
        #---------------------------------#
        #   对图像进行色域变换
        #   计算色域变换的参数
        #---------------------------------#
        r               = np.random.uniform(-1, 1, 3) * [hue, sat, val] + 1
        #---------------------------------#
        #   将图像转到HSV上
        #---------------------------------#
        hue, sat, val   = cv2.split(cv2.cvtColor(image_data, cv2.COLOR_RGB2HSV))
        dtype           = image_data.dtype
        #---------------------------------#
        #   应用变换
        #---------------------------------#
        x       = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        image_data = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        image_data = cv2.cvtColor(image_data, cv2.COLOR_HSV2RGB)

        #---------------------------------#
        #   对真实框进行调整
        #---------------------------------#
        if len(box)>0:
            np.random.shuffle(box)
            box[:, [0,2]] = box[:, [0,2]]*nw/iw + dx
            box[:, [1,3]] = box[:, [1,3]]*nh/ih + dy
            if flip: box[:, [0,2]] = w - box[:, [2,0]]
            box[:, 0:2][box[:, 0:2]<0] = 0
            box[:, 2][box[:, 2]>w] = w
            box[:, 3][box[:, 3]>h] = h
            box_w = box[:, 2] - box[:, 0]
            box_h = box[:, 3] - box[:, 1]
            box = box[np.logical_and(box_w>1, box_h>1)] 
        
        return image_data, box
    
# DataLoader中collate_fn使用
def yolo_dataset_collate(batch):
    images = []
    bboxes = []
    for img, box in batch:
        images.append(img)
        bboxes.append(box)
    images = torch.from_numpy(np.array(images)).type(torch.FloatTensor)
    bboxes = [torch.from_numpy(ann).type(torch.FloatTensor) for ann in bboxes]
    return images, bboxes

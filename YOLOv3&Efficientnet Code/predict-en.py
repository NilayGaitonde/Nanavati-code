#-----------------------------------------------------------------------#
#   predict.py integrates single image prediction, camera detection, FPS testing, and directory traversal detection
#   Modes can be modified by specifying the mode.
#-----------------------------------------------------------------------#
import time

import cv2
import numpy as np
from PIL import Image
import os
from yolo import YOLO

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
if __name__ == "__main__":
    yolo = YOLO()
    #----------------------------------------------------------------------------------------------------------#
    #   mode is used to specify the test mode:
    #   'predict'           Indicates single image prediction. If you want to modify the prediction process, such as saving images or cropping objects, you can refer to the detailed comments below
    #   'video'             Indicates video detection, can call camera or video for detection, see details in the comments below.
    #   'fps'               Indicates FPS testing, using street.jpg in the img folder, see details in the comments below.
    #   'dir_predict'       Indicates detection and saving by traversing the folder. Default traverses img folder, saves in img_out folder, see details in the comments below.
    #   'heatmap'           Indicates heat map visualization of prediction results, see details in the comments below.
    #----------------------------------------------------------------------------------------------------------#
    mode = "predict"
    #-------------------------------------------------------------------------#
    #   crop                Specifies whether to crop the target after single image prediction
    #   count               Specifies whether to count targets
    #   crop, count are only valid when mode='predict'
    #-------------------------------------------------------------------------#
    crop            = False
    count           = True
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Used to specify the video path, when video_path=0 it means detecting camera
    #                       To detect video, set video_path = "xxx.mp4", representing reading xxx.mp4 file in the root directory.
    #   video_save_path     Indicates the path to save the video, when video_save_path="" means no saving
    #                       To save video, set video_save_path = "yyy.mp4", representing saving as yyy.mp4 in the root directory.
    #   video_fps           Used for the fps of the saved video
    #
    #   video_path, video_save_path, and video_fps are only valid in mode='video'
    #   When saving video, you need to ctrl+c to exit or run to the last frame to complete the entire saving step.
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0
    #----------------------------------------------------------------------------------------------------------#
    #   test_interval       Used to specify the number of image detections when measuring fps. Theoretically, the larger test_interval, the more accurate fps.
    #   fps_image_path      Used to specify the fps test image
    #   
    #   test_interval and fps_image_path are only valid in mode='fps'
    #----------------------------------------------------------------------------------------------------------#
    test_interval   = 100
    fps_image_path  = "img/a.jpg"
    #-------------------------------------------------------------------------#
    #   dir_origin_path     Specifies the folder path of images to be detected
    #   dir_save_path       Specifies the save path for detected images
    #   
    #   dir_origin_path and dir_save_path are only valid in mode='dir_predict'
    #-------------------------------------------------------------------------#
    dir_origin_path = "img/"
    dir_save_path   = "img_out/"
    #-------------------------------------------------------------------------#
    #   heatmap_save_path   Save path for heat map, default saved in model_data
    #   
    #   heatmap_save_path is only valid in mode='heatmap'
    #-------------------------------------------------------------------------#
    heatmap_save_path = "model_data/heatmap_vision.png"
    

    if mode == "predict":
        '''
        1. If you want to save the detected image, use r_image.save("img.jpg"), modify directly in predict.py.
        2. If you want to get prediction box coordinates, go to yolo.detect_image function and read top, left, bottom, right values in the drawing part.
        3. If you want to crop the target using prediction boxes, go to yolo.detect_image function and use top, left, bottom, right values to crop using matrix method on the original image.
        4. If you want to write extra text on the prediction image, such as the number of specific detected targets, go to yolo.detect_image function, judge predicted_class in the drawing part,
        for example, judge if predicted_class == 'car': to determine if the current target is a car, then record the number. Use draw.text to write.
        '''
        while True:
            img = "/Users/nilaygaitonde/Documents/Projects/Nanavati/bleh.png"
            # img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                r_image = yolo.detect_image(image, crop = crop, count=count)
                r_image.show()

    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Failed to read camera (video) correctly. Please check if the camera is properly installed (or if the video path is correct).")

        fps = 0.0
        while(True):
            t1 = time.time()
            # Read a frame
            ref, frame = capture.read()
            if not ref:
                break
            # Format conversion, BGRtoRGB
            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)
            # Convert to Image
            frame = Image.fromarray(np.uint8(frame))
            # Perform detection
            frame = np.array(yolo.detect_image(frame))
            # RGBtoBGR to match opencv display format
            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)
            
            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff 
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()
        
    elif mode == "fps":
        img = Image.open(fps_image_path)
        tact_time = yolo.get_FPS(img, test_interval)
        print(str(tact_time) + ' seconds, ' + str(1/tact_time) + 'FPS, @batch_size 1')

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = yolo.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)

    elif mode == "heatmap":
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                continue
            else:
                yolo.detect_heatmap(image, heatmap_save_path)
        
    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video', 'fps', 'heatmap', 'dir_predict'.")
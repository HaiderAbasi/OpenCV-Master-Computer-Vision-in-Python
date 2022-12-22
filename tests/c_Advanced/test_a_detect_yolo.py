import unittest
import cv2
import glob
import os
from loguru import logger


from src.c__Advanced.Object_detection.b_Yolo.a_detect_yolo import Dnn
from src.utilities import get_iou,to_ltrd



class TestDetectYolo(unittest.TestCase):
        
    def test_detectyolo(self):
        print('''[Testing]: Evaluating trained yolo-v7-tiny on a test (road-signs) dataset.
              \n Remember You ned 70% .accuracy to pass!. No Pressure :)\n''')
        
        # Create your inputs
        
        # Getting required obj-detection files [Model,Weight,Classes]
        Model_dir = "Data/dnn/yolo_v7/road-signs"
        if (os.path.isdir(Model_dir)):
            model_path   = os.path.join(Model_dir,"yolov7-tiny_road_signs_TL.cfg")
            weights_path = os.path.join(Model_dir,"yolov7-tiny_road_signs_TL_best.weights")
            classes_path = os.path.join(Model_dir,"road_signs.names")
        else:
            training_in_colab = "https://colab.research.google.com/drive/1ZsbiV62151gw2qwI3pdkqV3nojCx1T6x?usp=sharing#scrollTo=AqQaECSLJ8Yv"
            logger.error(f"\n\n> Train the yoloV7-tiny using the method described in...\n\n {training_in_colab}\n\n - traffics-sign dataset you will use...\
                            \n\n    https://universe.roboflow.com/roboflow-100/road-signs-6ih4y/browse?queryText=&pageSize=50&startingIndex=0&browseQuery=true\n")
            exit()

        # Loading the Yolo-v7-tiny trained on a traffic-signs for testing on a video.
        detection_yolo = Dnn(model_path,weights_path,classes_path,0.4)

        # gt_annotations [annotations in 10 images] 
        #                [1st image annotation,   2nd image annotation    ,....]
        #                [[1 2 5 5]["stop"]   ,[1 2 5 5]["speed_sign_20"] ,....]
        # gt_annotations = [
        #                   [[787, 119, 37, 78],"traffic-light"] , [[1301, 275, 104, 107],"stop"]         , [[1301, 275, 104, 107],"stop"],
        #                   [[285, 239, 54, 72],"speed_sign_60"] , [[169, 382, 141, 164],"speed_sign_40"] , [[455, 282, 54, 61],"speed_sign_60"],
        #                   [[822, 378, 45, 43],"stop"]          , [[505, 180, 39, 71],"traffic-light"]   , [[1042, 246, 58, 44],"speed_sign_40"],
        #                   [[1042, 246, 58, 44],"speed_sign_40"], [[773, 367, 35, 33],"stop"]            , [[7, 28, 107, 104],"cross-walk"]
        #                  ]
        gt_annotations = [
                          [[740, 126, 32, 8],"traffic-light"]    , [[1106, 179, 97, 65],"stop"]           ,[[265, 252, 55, 56],"speed_sign_60"] ,
                          [[170, 415, 106, 140],"speed_sign_40"] , [[423, 299, 52, 59],"speed_sign_60"]   ,[[776, 354, 38, 40],"stop"]          ,
                          [[473, 168, 36, 68],"traffic-light"]   , [[960, 259, 42, 62],"speed_sign_40"]   ,[[959, 259, 45, 61],"stop"]          ,
                          [[45, 109, 490, 396],"cross-walk"]
                         ]
        
        # Fetch image paths from directory
        exts = ['*.png', '*.jpg']
        image_list = [f for ext in exts for f in glob.glob(os.path.join("tests/fixtures/advanced/yolo", ext))]
             
        # Contatining for identifying the correct predictions by idx
        pred_matched = [0] * len(image_list)

        # Predicting objects using trained model and comparing with ground truth.
        for idx, img_path in enumerate(image_list):
            # Reading current image
            test_img = cv2.imread(img_path,cv2.IMREAD_COLOR)
            # Resizing to a default size
            test_img = cv2.resize(test_img,(1280,720))

            # Fetching ground truth predictions
            gt_bbox    , gt_class     = gt_annotations[idx]
            x,y,w,h= gt_bbox
            # Performing detection using trained model on the test images
            pred_bboxes, pred_classes = detection_yolo.detect(test_img)
            
            cv2.rectangle(test_img,(x,y),(x+w,y+h),(0,255,0),2)
            cv2.namedWindow(f"{idx}: Model predicted -> {pred_classes} ",cv2.WINDOW_NORMAL)
            cv2.imshow(f"{idx}: Model predicted -> {pred_classes} ",test_img)
            cv2.waitKey(0)
            cv2.destroyWindow(f"{idx}: Model predicted -> {pred_classes} ")

            # Looping over all the predictions in the curr_img to see if any one was matching the GT
            for i,pred_bbox in enumerate(pred_bboxes):
                pred_class = pred_classes[i]
                # Estimating the iou between pred and Ground truth
                iou = get_iou(to_ltrd(pred_bbox),to_ltrd(gt_bbox))
                print(iou)
                # If significant overlap and correct predicted class.
                #    Then indicate in pred_matched at the respective index
                if iou > 0.5:
                    # Overlap, Check Class
                    if pred_class == gt_class:    
                        pred_matched[idx] = 1

        # Estimate how many you get correct from the total
        detection_accuracy = sum(pred_matched)/len(pred_matched)

        # Should be greater than 6! 
        # Otherwise > Further/Re Training is Needed!
        self.assertGreater(detection_accuracy,0.6,msg = "\n\n [Error]:\n\n   >>> Model either incorrectly/incompletely trained <<< \n")
        
if __name__ == "__main__":
    unittest.main()
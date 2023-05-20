import unittest
import cv2
import os

from src.b__CV_101.b_image_filtering import assignment

from tests.utils import download_missing_test_data




class TestImageFiltering(unittest.TestCase):
        
    def test_imagefiltering(self):
        
        print('''\n[Testing]: Checking Meteor (ROI) was properly highlighted in the test-video''')
        
        # Create your inputs
        ref_vid = cv2.VideoCapture(r"tests\fixtures\meteor_mini.mp4")
        ref_crop = cv2.imread(r"tests\fixtures\meteor_caught.png",cv2.IMREAD_UNCHANGED)
        
        ref_crop_img = ref_crop[:,:,:3]
        ref_crop_mask = ref_crop[:,:,3]
        
        ref_crop_mask_inv = cv2.bitwise_not(ref_crop_mask)
        
        #ref_meteor = cv2.bitwise_and(ref_crop_img,ref_crop_img,mask = ref_crop_mask)
        #ref_meteor_less = cv2.bitwise_and(ref_crop_img,ref_crop_img,mask = ref_crop_mask_inv)
        
        if not os.path.isfile(r"src\b__CV_101\vid_roi_highlighted.avi"):
            # Execute assignment to save roi-highlighted video to disk
            assignment(debug=False)
        
        # Read in the videocapture object for result video
        result_vid = cv2.VideoCapture(r"src\b__CV_101\vid_roi_highlighted.avi")
        
        frame_no = 0
        while(ref_vid.isOpened()):
            ref_ret,ref_frame = ref_vid.read()
            ret,frame = result_vid.read()
            if ref_ret:
                if frame_no ==240:
    
                    # 4) Adding scaled mask to orig image to perform unsharp maskening
                    diff = cv2.addWeighted(frame, 1.0, ref_frame, -1, 0)
                    
                    crop = diff[100:300,500:700]
                    crop_gray = cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
                    mask = cv2.threshold(crop_gray,0,255,cv2.THRESH_BINARY|cv2.THRESH_OTSU)[1]

                    mask_meteor = mask.copy()
                    mask_meteorless = cv2.bitwise_not(mask_meteor)

                    perc_common = ((mask_meteor == ref_crop_mask).sum())/(mask_meteor.size)
                    perc_common_inv = ((mask_meteorless == ref_crop_mask_inv).sum())/(mask_meteor.size)

                    
                    self.assertGreater(perc_common,0.97,msg = "\n\n [Error]:\n\n   >>> Required Target(Highlighted-ROI) not achieved! <<< \n")
                    self.assertGreater(perc_common_inv,0.90,msg = "\n\n [Error]:\n\n   >>> Only the ROI needed to be highlighted <<< \n")
                    
                    break
                
                
                frame_no = frame_no + 1
                
        result_vid.release()
        
        

if __name__ == "__main__":
    download_missing_test_data()
    unittest.main()
    
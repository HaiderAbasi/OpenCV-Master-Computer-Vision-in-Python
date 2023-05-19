import unittest
from src.a__IP_Basics.d_trackbar import assignment
import cv2

from tests.utils import Helper
from loguru import logger

from tests.utils import download_missing_test_data

class TestTrackbar(unittest.TestCase):
        
    def test_trackbar(self):
        print('''\n[Testing]: Checking if messi-img has been returned with the football turned blue or not!\n''')
        logger.warning("> Disclaimer: This tests requires user-input. Without which it cannot be carried out!\n")

        debug = False
        
        # Create your inputs
        blue_ftball = cv2.imread("tests/fixtures/messi_bluefootball.png",cv2.IMREAD_UNCHANGED)
        ref_img = blue_ftball[:,:,:3]
        ref_mask = blue_ftball[:,:,3]
        
        
        ref_ball = cv2.bitwise_and(ref_img,ref_img,mask = ref_mask)

        result = assignment(debug)
        
        result_ball = cv2.bitwise_and(result,result,mask = ref_mask)

        if debug:
            cv2.imshow("result_ball",result_ball)
            cv2.imshow("ref_ball",ref_ball)
            cv2.waitKey(0)
            
        hlpr = Helper()
        img_ref_error_perc = hlpr.is_largely_close(result,ref_img)
        self.assertLess(img_ref_error_perc,6,msg="\n\n [Suggestion]:\n\n   >>> Only Modify the ball <<< \n")
        
        ball_ref_error_perc = hlpr.is_largely_close(result_ball,ref_ball)
        self.assertLess(ball_ref_error_perc,1,msg="\n\n [Suggestion]:\n\n   >>> Not blue enough <<< \n")

if __name__ == "__main__":
    download_missing_test_data()
    unittest.main()
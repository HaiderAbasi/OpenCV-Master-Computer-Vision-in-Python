import unittest
import cv2
from src.b__CV_101.e_Image_features import find_obj_inscene
from src.a__IP_Basics.e_mouse_events import assignment

from loguru import logger

from tests.utils import download_missing_test_data

class TestMouseEvents(unittest.TestCase):
        
    def test_trackbar(self):
        print('''\n[Testing]: Return the (Upscaled)-Second image by navigating and resizing inside the Image Viewer\n''')
        logger.warning("> Disclaimer: This tests requires user-input. Without which it cannot be carried out!\n")

        # Create your inputs
        ref_img = cv2.imread(r"tests\fixtures\me_img_upscaled.png",cv2.IMREAD_UNCHANGED)

        result = assignment(debug = False)

        result[:130,:] = 0
        ref_img[:130,:] = 0
        
        # 1) Match the obj to scene. See if we are on correct image
        M = find_obj_inscene(result,ref_img,"sift",False)
        self.assertIsNotNone(M,msg= "\n\n [Error]:\n\n   >>> Incorrect Image selected! <<< \n")

        # Our Picture Viewer Default Window Size
        WIDTH = 480
        HEIGHT = 360
        
        ref_area = HEIGHT*WIDTH
        rows,cols = result.shape[0:2]
        area = rows*cols
        
        self.assertGreater(area,ref_area,msg = "\n\n [Error]:\n\n   >>> Image not upsized! <<< \n")

if __name__ == "__main__":
    download_missing_test_data()
    unittest.main()
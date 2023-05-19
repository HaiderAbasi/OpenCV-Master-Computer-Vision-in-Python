import unittest
import numpy as np
import cv2
from src.a__IP_Basics.c_imp_functions import assignment
from src.b__CV_101.e_Image_features import find_obj_inscene

from loguru import logger

from tests.utils import download_missing_test_data

class TestImpFunctions(unittest.TestCase):
        
    def test_impfunctions(self):
        print('''\n[Testing]: Checking if bottom-left plant has been returned without shadow!\n''')
        logger.warning("> Disclaimer: This tests requires user-input. Without which it cannot be carried out!\n")

        # Create your inputs
        Plant_no_shadow = cv2.imread("tests/fixtures/Plant-no-shadow.png",cv2.IMREAD_UNCHANGED)

        result = assignment(debug=False)

        self.assertEqual(len(result.shape),2,msg= "\n\n [Error]:\n\n   >>> Expected ROI to be single channel <<< \n")
        # 1) Match the obj to scene. Find its homography
        M_orig = find_obj_inscene(result,Plant_no_shadow,"sift",False,10)
        self.assertIsNotNone(M_orig,msg= "\n\n [Error]:\n\n   >>> Incorrect/No plant selected! <<< \n")
        
        # 2) Transform the distorted obj to match the orthomosaic matching location.
        ref_rows,ref_cols = Plant_no_shadow.shape[0:2]
        ref_area = ref_rows*ref_cols
        rows,cols = result.shape[0:2]
        area = rows*cols

        if not (np.isclose(area,ref_area,3e-1)):
            self.assertLess(area,ref_area,msg = "\n\n [Suggestion]:\n\n   >>> Try restricting to only the ROI <<< \n")
            self.assertGreater(area,ref_area,msg = "\n\n [Suggestion]:\n\n   >>> Try covering complete ROI <<< \n")

            
        
        
        
if __name__ == "__main__":
    download_missing_test_data()
    unittest.main()
import unittest
import cv2

from src.b__CV_101.a_image_transformations import assignment
from src.b__CV_101.e_Image_features import find_obj_inscene

import numpy as np

from tst.utils import download_missing_test_data


class TestImageTransformations(unittest.TestCase):
        
    def test_imagetransformations(self):
        print('''\n[Testing]: Return the Upscaled second image by navigating and resizing inside the Image Viewer''')
        
        # Create your inputs
        ref_img = cv2.imread(r"tst\fixtures\bookcover.png",cv2.IMREAD_UNCHANGED)

        result = assignment(debug = False)
        
        # 1) Match the obj to scene. See if we are on correct image
        M = find_obj_inscene(result,ref_img,"sift",False)
        self.assertIsNotNone(M,msg= "\n\n [Error]:\n\n   >>> Incorrect Object selected! <<< \n")

        M_ref = np.eye(M.shape[0],M.shape[1])
        
        np.testing.assert_allclose(M,M_ref,atol=20,err_msg= "\n\n [Error]:\n\n   >>> Final image should have only book cover (Nothing-else) <<< \n")
        
        
if __name__ == "__main__":
    download_missing_test_data()
    unittest.main()
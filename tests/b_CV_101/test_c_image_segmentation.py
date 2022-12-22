import unittest
from src.b__CV_101.c_image_segmentation import assignment
import cv2
import numpy as np

class TestImageSegmentation(unittest.TestCase):
        
    def test_imagesegmentation(self):
        print('''\n[Testing]: Checking if Segmentation assignment was completed correctly..\n''')
        
        # Create your inputs
        ref_img = cv2.imread(r"tests\fixtures\segmented_plants.png",cv2.IMREAD_UNCHANGED)
        ref_mask = ref_img[:,:,3]

        result = assignment(debug = False)
        result_mask = result[:,:,3]

        perc_common = ((result_mask == ref_mask).sum())/(result_mask.size)

        self.assertGreater(perc_common,0.97,msg = "\n\n [Error]:\n\n   >>> Adequate No of plants not segmented <<< \n")

        
        

if __name__ == "__main__":
    unittest.main()
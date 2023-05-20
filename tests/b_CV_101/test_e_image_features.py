import unittest
import cv2
import numpy as np

from src.b__CV_101.e_Image_features import assignment,find_obj_inscene

from tests.utils import download_missing_test_data


class TestImageFeatures(unittest.TestCase):
        
    def test_imagefeatures(self):
        
        print('''[Testing]: Testing if img-registration (assignment) is working as required or not!\n''')
        
        # Create your inputs
        ref_transformed = cv2.imread(r"tests\fixtures\drone_view_transformed.png",cv2.IMREAD_UNCHANGED)
        
        ref_scene = cv2.imread(r"tests\fixtures\scene_obj_removed.png",cv2.IMREAD_UNCHANGED)
        ref_scene_roi = ref_scene[:,:,:3]
        ref_scene_mask = ref_scene[:,:,3]

        result = assignment(debug = False)
        
        # 1) Match the obj to scene. See if we are on correct image
        M = find_obj_inscene(result,ref_transformed,"sift",debug = False)
        self.assertIsNotNone(M,msg= "\n\n [Error]:\n\n   >>> Incorrect Object selected! <<< \n")
        M_ref = np.eye(M.shape[0],M.shape[1])
        
        # [Test]: Checking if drone-view properly transformed or not
        np.testing.assert_allclose(M,M_ref,atol=1,err_msg= "\n\n [Error]:\n\n   >>> drone-view not transformed correctly <<< \n")
        
        # [Test]: Check if complete Image registration has been done.
        result_roi = cv2.bitwise_and(result,result,mask = ref_scene_mask)
        np.testing.assert_allclose(result_roi,ref_scene_roi,atol=5,err_msg= "\n\n [Error]:\n\n   >>> Transformed drone-view not registered on map <<< \n")

        
    
    
    
if __name__ == "__main__":
    download_missing_test_data()
    unittest.main()
    
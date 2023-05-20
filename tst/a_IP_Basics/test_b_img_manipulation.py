import unittest
from src.a__IP_Basics.b_img_manipulation import assignment
import numpy as np
import cv2

from tst.utils import download_missing_test_data


class TestImgManipulation(unittest.TestCase):
        
    def test_imgmanipulation(self):
        print('''\n[Testing]: Checking if solar eclipse has been correctly visualized or not!\n''')

        # Create your inputs
        eclipse = cv2.imread("tests/fixtures/eclipse.png")
        
        result = assignment(debug= False)
        

        sun_loc = ( int(eclipse.shape[1]/2) + 150 , int(eclipse.shape[0]/2) )
        img_m_w = 300
        img_m_h = 300
            
        eclipse_crop = eclipse[sun_loc[1]-int(img_m_w/2):sun_loc[1]+int(img_m_w/2),
                             sun_loc[0]-int(img_m_h/2):sun_loc[0]+int(img_m_h/2)]
        
        result_crop = result[sun_loc[1]-int(img_m_w/2):sun_loc[1]+int(img_m_w/2),
                             sun_loc[0]-int(img_m_h/2):sun_loc[0]+int(img_m_h/2)]
        

        np.testing.assert_allclose(result_crop,eclipse_crop,err_msg="\n\n [Suggestion]:\n >>> Solar eclipse is when the moon comes between sun and earth! <<< \n")
        
        np.testing.assert_allclose(result,eclipse,err_msg="\n\n [Suggestion]:\n\n   >>> Wouldn't it be more dark & gloomy if the sun got blocked by a big darn ball? <<< \n")

        
        
        
if __name__ == "__main__":
    download_missing_test_data()
    unittest.main()
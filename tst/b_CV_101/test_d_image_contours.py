import unittest
from src.b__CV_101.d_image_contours import assignment
from src.utilities import closest_node,euc_dist

from tst.utils import download_missing_test_data



class TestImageContours(unittest.TestCase):
        
    def test_imagecontours(self):
        print('''[Testing]: Checking if desired number of shapes and centers were determined or not\n''')
        
        # Reference inputs
        ref_shapes = ['square', 'rectangle', 'pentagon', 'triangle', 'rectangle', 'rectangle', 'rectangle']
        ref_cntrs = [(304, 207), (294, 79), (77, 215), (138, 106), (313, 206), (135, 115), (130, 144)]

        result_shapes,result_cntrs = assignment(debug = False)

        correct_shapes = []
        correct_cntrs = []
        for i,shape in enumerate(result_shapes):
            cntr = result_cntrs[i]
            clst_pt_idx = closest_node(cntr,ref_cntrs)
            clst_ref_cntr = ref_cntrs[clst_pt_idx]
            if ((euc_dist(cntr,clst_ref_cntr)) < 10):
                # Valid Location. Lets confirm shape
                if (shape == ref_shapes[clst_pt_idx]):
                    # If pred shape is actual shape
                    # Append shape and its centers to correct shapes
                    correct_shapes.append(shape)
                    correct_cntrs.append(cntr)
        
        self.assertGreater(len(correct_shapes),int(0.70*len(ref_shapes)),msg = "\n\n [Error]:\n\n   >>> Identify atleast 5 shapes correctly <<< \n")

        
        

if __name__ == "__main__":
    download_missing_test_data()
    unittest.main()
    
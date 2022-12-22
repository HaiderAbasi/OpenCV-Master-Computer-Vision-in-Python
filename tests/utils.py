import numpy as np




class Helper():
    @staticmethod
    def is_largely_close(cmp,ref,error_margin = 10):
        diff_mask = abs(ref - cmp) > error_margin
        ref_error_perc = (np.count_nonzero(diff_mask)/diff_mask.size)*100
        return ref_error_perc
    
            
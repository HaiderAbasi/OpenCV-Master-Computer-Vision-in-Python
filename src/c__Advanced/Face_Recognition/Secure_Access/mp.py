import cv2
import time
# Multoprocessing imports
from multiprocessing import Array
import concurrent.futures
# Pickle solution imports
import copyreg as copy_reg
import types

def _pickle_method(m):
    if m.im_self is None:
        return getattr, (m.im_class, m.im_func.func_name)
    else:
        return getattr, (m.im_self, m.im_func.func_name)

copy_reg.pickle(types.MethodType, _pickle_method)


class Testing():

    def __init__(self):
        # Creating an object (executor) of the process pool executor in concurrent.futures for multiprocessing
        self.executor = concurrent.futures.ProcessPoolExecutor()
        self.futures = None
        # To share data acroos multiple processes we need to create a shared adress space. 
        # Here we use multiprocessing.Array which stores an Array of c data types and can be shared along multiple processes
        self.shared_array = Array('i', 4) # Array of integer type of length 4 ==> Used here for sharing the computed bbox
    
    def wait_n_secs(self,n):
        print(f"I wait for {n} sec")
        cv2.waitKey(n*1000)
        wait_array = (n,n,n,n)
        return wait_array
    
def function(waittime):
    bbox = Testing().wait_n_secs(waittime)
    return bbox

def wrapper_func(ins,*args):
    ins.wait_n_secs(args)


if __name__ =="__main__":
    
    # Creating Object of Testing Class
    testing = Testing()
    
    waittime = 5
    # Not working!
    #testing.futures = testing.executor.submit(testing.wait_n_secs,waittime) # Instance Method 
    # Working!
    #testing.futures = testing.executor.submit(function,waittime) # function calling instance method
    inp_args = [waittime]
    testing.futures = testing.executor.submit(wrapper_func,testing,inp_args) # function calling instance method

    stime = time.time()
    while 1:
        if not testing.futures.running():
            print("Checking for results")
            testing.shared_array = testing.futures.result()
            print("Shared_array received = ",testing.shared_array)
            break
        time_elapsed = time.time()-stime
        if (( time_elapsed % 1 ) < 0.001):
            print(f"Time elapsed since some time = {time_elapsed:.2f} sec")
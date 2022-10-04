import time
from multiprocessing import Array
import concurrent.futures
# get all active child processes for the main process with no children
from multiprocessing import active_children
import multiprocessing
class Testing():

    def __init__(self):
        self.executor = concurrent.futures.ProcessPoolExecutor()
        self.futures = None
        self.shared_array = Array('i', 4)

    def wait_n_secs(self, n):
        print(f"I wait for {n} sec")

        # Have your own implementation here
        time.sleep(n)
        wait_array = (n, n, n, n)
        return wait_array

    def __getstate__(self):
        d = self.__dict__.copy()

        # Delete all unpicklable attributes.
        del d['executor']
        del d['futures']
        del d['shared_array']

        return d


if __name__ == "__main__":

    testing = Testing()

    waittime = 5
    testing.futures = testing.executor.submit(testing.wait_n_secs, waittime)

    stime = time.time()
    while 1:
        if not testing.futures.running():
            print("Checking for results")
            testing.shared_array = testing.futures.result()
            print("Shared_array received = ", testing.shared_array)
            break
        time_elapsed = time.time() - stime
        if ((time_elapsed % 1) < 0.001):
            print(f"Time elapsed since some time = {time_elapsed:.2f} sec")
        # get all active child processes
        children = active_children()
        print(f"{len(children)} subprocess running!")
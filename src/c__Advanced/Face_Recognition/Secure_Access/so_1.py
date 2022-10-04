import time
from multiprocessing import Array
import concurrent.futures


class Testing():

    def __init__(self):
        self.executor = concurrent.futures.ProcessPoolExecutor()
        self.futures = None
        self.shared_array = Array('i', 4)

    @staticmethod
    def wait_n_secs(n):
        print(f"I wait for {n} sec")

        # Have your own implementation here
        time.sleep(n)
        wait_array = (n, n, n, n)
        return wait_array


if __name__ == "__main__":

    testing = Testing()

    waittime = 5

    #print("type(testing) = ",type(testing))
    #testing.futures = testing.executor.submit(type(testing).wait_n_secs, waittime)  # Notice the type(testing)
    testing.futures = testing.executor.submit(testing.wait_n_secs, waittime)  # Notice the type(testing)

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
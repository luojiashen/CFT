from collections import defaultdict
from time import time, sleep
class Timer:
    def __init__(self):
        self.process_time_gaps = defaultdict(list)
        self.process_timestemp = defaultdict(int)
        self.running_process = list()
    
    def start(self, process_name):
        self.process_timestemp[process_name] = time()
        self.running_process.append(process_name)

    def stop(self, process_name):
        self.process_time_gaps[process_name] = time() - self.process_timestemp[process_name]
        self.running_process.pop(self.running_process.index(process_name))

    def stop_all(self):
        time_now = time()
        for p in self.running_process:
            self.process_time_gaps[p] = time_now - self.process_timestemp
        self.running_process = list()

    def __call__(self):
        return self.process_time_gaps
    
if __name__ == "__main__":
    t = Timer()
    print(t.process_timestemp)
    t.start("process1")
    print("running process:", t.running_process)
    sleep(1)
    t.start("process2")
    print("running process:", t.running_process)
    sleep(0.5)
    t.stop("process1")
    print("running process:", t.running_process)
    t.stop("process2")
    print("running process:", t.running_process)
    print(t())
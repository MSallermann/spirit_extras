import psutil
from time import sleep, perf_counter_ns
from datetime import datetime
import os
import matplotlib.pyplot as plt
import numpy as np
import threading

class MemoryMonitor:
    def __init__(self, dt=0.1):
        self.keep_measuring = True
        self.current_usage  = []
        self.timestamps     = []
        self.time_list      = []
        self.dt             = dt
        self.time           = datetime.now()
        self.max_usage      = 0
        self.t0             = 0
        self.mem0           = 0

    def _snapshot(self):
        process = psutil.Process(os.getpid())
        mem = process.memory_full_info()[0] / float(2 ** 20) # MB
        self.current_usage.append(mem - self.mem0)
        self.max_usage = max(
            self.max_usage, mem
        )
        self.time_list.append(perf_counter_ns() - self.t0)

    def _measure_usage(self):
        self.t0 = perf_counter_ns()
        self.time_list = []
        self.keep_measuring = True
        while self.keep_measuring:
            self._snapshot()
            sleep(self.dt)

    def plot(self, file):
        self.keep_measuring = False
        fig, ax = plt.subplots()
        for ts in self.timestamps:
            ax.axvline(ts[0]*1e-9, color="black", ls="--")
            ax.text(ts[0]*1e-9, (self.max_usage-self.mem0) * 1, ts[1], verticalalignment="center", horizontalalignment="center", rotation=90, bbox=dict(facecolor = "white"))

        ax.plot(np.asarray(self.time_list) * 1e-9, self.current_usage)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("Memory [MB]")
        fig.savefig(file, dpi=100)
        return self.max_usage

    def start(self):
        self.monitor_thread = threading.Thread(target=self._measure_usage, args=(), daemon=True)
        self.monitor_thread.start()

    def stop(self):
        self._snapshot()
        self.keep_measuring = False
        self.monitor_thread.join()

    def to_file(self, filename):
        data = np.zeros(shape = (len(self.time_list), 2))
        data[:,0] = self.time_list
        data[:,1] = self.current_usage
        np.savetxt(filename, data, header = "time [ns], memory_usage [MB]")

    def __enter__(self):
        self.start()
        return self

    def __exit__(self, type, value, traceback):
        self.stop()
        return False

    def start_from_here(self):
        self._snapshot()
        self.t0            = perf_counter_ns()
        self.mem0          = self.current_usage[-1]
        self.time_list     = [0]
        self.current_usage = [0]

    def timestamp(self, label=""):
        self.timestamps.append( [perf_counter_ns()-self.t0, label] )
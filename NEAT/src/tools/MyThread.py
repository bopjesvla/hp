import threading

from tools.DEBUG import *

class MyThread(threading.Thread):
    def __init__(self, threadID, name, function):
        threading.Thread.__init__(self)
        self.threadID = threadID
        self.name     = name
        self.function = function

    def run(self):
        dprt(self,"Starting Thread " + self.name)
        self.function()
        dprt(self,"Closing  Thread " + self.name)

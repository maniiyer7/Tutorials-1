#!/usr/local/bin/python
###############################################################################
__author__ = 'amirkavousian'
__email__ = 'amir.kavousian@sunrun.com'
# created on: April 26, 2015
# summary: multi-threading, multi-processing playground
###############################################################################

import threading
import multiprocessing
import time, os, sys, inspect, logging, datetime, subprocess, pytz, ConfigParser
from multiprocessing import Process, Queue, Manager

###############################################################################
# The two main modules for parallel processing in Python are 'multiprocessing' and 'threading'

# The 'threading' module does not use multi-core threading.
# Since CPython doesn't use multiple cores to run CPU-bound tasks anyway, the only reason for using 'threading' is not blocking the process while there's a wait for some I/O.
# If you want to benefit from multiple cores for CPU-bound tasks, use 'multiprocessing'.


# The threading module uses threads, the multiprocessing uses processes.
# The difference is that threads run in the same memory space, while processes have separate memory.
# This makes it a bit harder to share objects between processes with multiprocessing.
# Since threads use the same memory, precautions have to be taken or two threads will write to the same memory at the same time.
# This is what the global interpreter lock is for.
# Spawning processes is a bit slower than spawning threads. Once they are running, there is not much difference.

# Functionality within this package requires that the __main__ module be importable by the children.
# This is covered in Programming guidelines however it is worth pointing out here.
# This means that some examples, such as the Pool examples will not work in the interactive interpreter.


### USEFUL LINKS:
# http://stackoverflow.com/questions/2846653/python-multithreading-for-dummies/28463266#28463266
# http://chriskiehl.com/article/parallelism-in-one-line/

###############################################################################
# PyCharm: note that the Python console in PyCharm is an interactive console.
# This causes issues with starting a new process in PyCharm, when using this interactive console.
# Try running the script from Terminal (either through the Terminal programs or through the Terminal tab in PyCharm).
# http://stackoverflow.com/questions/24171725/scikit-learn-multicore-attributeerror-stdin-instance-has-no-attribute-close
# http://stackoverflow.com/questions/10948025/multiprocessing-attributeerror-stdin-instance-has-no-attribute-close
###############################################################################

###############################################################################
################################# threading ###################################
###############################################################################

###############################################################################
### USING THREADING MODULE
# This is a case where threading is used as a simple optimization:
# each subthread is waiting for a URL to resolve and respond, in order to put its contents on the queue;
# each thread is a daemon (won't keep the process up if main thread ends -- that's more common than not);
# the main thread starts all subthreads, does a get on the queue to wait until one of them has done a put,
# then emits the results and terminates (which takes down any subthreads that might still be running, since they're daemon threads).

# We are more interested in multiprocessing
if (False):
    import Queue
    import threading
    import urllib2

    # called by each thread
    def get_url(q, url):
        q.put(urllib2.urlopen(url).read())

    theurls = ["http://google.com", "http://yahoo.com"]

    print theurls

    q = Queue.Queue()

    for u in theurls:
        t = threading.Thread(target=get_url, args = (q,u))
        t.daemon = True
        t.start()

    s = q.get()
    print s
###############################################################################

###############################################################################
############################# multiprocessing #################################
###############################################################################

#######################################
### POOL (object in multiprocessing module)
# Pool object (in multiprocessing module) offers a convenient means of parallelizing the
# execution of a function across multiple input values,
# distributing the input data across processes (data parallelism).

# One can create a pool of processes which will carry out tasks submitted to it with the Pool class.
# A process pool object which controls a pool of worker processes to which jobs can be submitted.
# It supports asynchronous results with timeouts and callbacks and has a parallel map implementation.

from multiprocessing import Pool

if (False):
    def f(x):
        return x*x

    if __name__ == '__main__':
        p = Pool(5)
        print(p.map(f, [1, 2, 3]))
#######################################

#######################################
### PROCESS (object in multiprocessing module)
# In multiprocessing, processes are spawned by creating a Process object and then calling its start() method.
# Process objects represent activity that is run in a separate process.
# .start() : This must be called at most once per process object.
# .join() : Block the calling thread until the process whose join() method is called terminates or until the optional timeout occurs.

from multiprocessing import Process

if (False):
    def f(name):
        print 'hello', name

    if __name__ == '__main__':
        p = Process(target=f, args=('bob',))
        p.start()
        p.join()
#######################################

#######################################
### QUEUE: is a way of exchanging objects between processes
# The Queue class is a near clone of Queue.Queue.
# Queue uses a pipe and a few locks/semaphores.
# We can also create a shared queue by using a Manager object.

from multiprocessing import Process, Queue

if (False):
    def f(q):
        q.put([42, None, 'hello'])

    if __name__ == '__main__':
        q = Queue()
        p = Process(target=f, args=(q,))
        p.start()
        print q.get()    # prints "[42, None, 'hello']"
        p.join()
#######################################

#######################################
### PIPE: The Pipe() function returns a pair of connection objects connected by a pipe which by default is duplex (two-way).
# Returns a pair (conn1, conn2) of Connection objects representing the ends of a pipe.
# The two connection objects returned by Pipe() represent the two ends of the pipe. Each connection object has send() and recv() methods (among others).
# Since the two ends are connected, we can send data from one end, and receive it at the other end. Example below shows an example of that.

# Note that data in a pipe may become corrupted if two processes (or threads) try to read from or write to the same end of the pipe at the same time.
# Of course there is no risk of corruption from processes using different ends of the pipe at the same time.

# For passing messages one can use Pipe() (for a connection between two processes) or a queue (which allows multiple producers and consumers).

from multiprocessing import Process, Pipe

if (False):
    def f(conn):
        conn.send([43, None, 'hello Pipe'])  # send data through one end of the pipe.
        conn.close()

    if __name__ == '__main__':
        parent_conn, child_conn = Pipe()
        p = Process(target=f, args=(child_conn,))
        p.start()
        print parent_conn.recv()  # receive the data from the other end of the pipe, even though the data was sent by the other end in a separate process.
        p.join()
#######################################

#######################################
### LOCK
# multiprocessing contains equivalents of all the synchronization primitives from threading.
# For instance one can use a lock to ensure that only one process prints to standard output at a time.
from multiprocessing import Process, Lock

if (False):
    def f(l, i):
        l.acquire()
        print 'hello world', i
        l.release()

    if __name__ == '__main__':
        lock = Lock()

        for num in range(10):
            Process(target=f, args=(lock, num)).start()
            time.sleep(0.1)
#######################################

#######################################
### SHARING STATE BETWEEN PROCESSES
# Data can be stored in a shared memory map using Value or Array.
# The 'd' and 'i' arguments used when creating num and arr are typecodes of the kind used by the array module:
# 'd' indicates a double precision float and 'i' indicates a signed integer.

# The issue is that you cannot assign a value to a variable in a separate process,
# and then retrieve that later in the main process. You need to use a special object
# such as the Array or Value objects here to move them across different processes.
# The other alternative to using Array and Value is to use the Manager class (see below).

from multiprocessing import Process, Value, Array

if (True):
    def f(n, a):
        n.value = 3.1415927
        for i in range(len(a)):
            a[i] = -a[i]

    if __name__ == '__main__':
        num = Value('d', 0.0)
        arr = Array('i', range(10))

        p = Process(target=f, args=(num, arr))
        p.start()
        p.join()

        print num.value
        print arr[:]
#######################################

#######################################
### MANAGER
# A manager object returned by Manager() controls a server process which holds Python objects and allows other processes to manipulate them using proxies.
# A manager returned by Manager() will support types list, dict, Namespace, Lock, RLock, Semaphore, BoundedSemaphore, Condition, Event, Queue, Value and Array.
# Server process managers are more flexible than using shared memory objects because they can be made to support arbitrary object types.
# Also, a single manager can be shared by processes on different computers over a network.
# They are, however, slower than using shared memory.

# Managers provide a way to create data which can be shared between different processes.
# A manager object controls a server process which manages shared objects.
# Other processes can access the shared objects by using proxies.

from multiprocessing import Process, Manager

if (False):
    def f(d, l):
        d[1] = '1'
        d['2'] = 2
        d[0.25] = None
        l.reverse()

    if __name__ == '__main__':
        manager = Manager()

        d = manager.dict()
        l = manager.list(range(10))

        p = Process(target=f, args=(d, l))
        p.start()
        p.join()

        print d
        print l
#######################################

#######################################
### POOL OF WORKERS
# The Pool class represents a pool of worker processes.
from multiprocessing import Pool

if (False):
    def f(x):
        return x*x

    if __name__ == '__main__':
        pool = Pool(processes=4)              # start 4 worker processes
        result = pool.apply_async(f, [10])    # evaluate "f(10)" asynchronously
        print result.get(timeout=1)           # prints "100" unless your computer is *very* slow
        print pool.map(f, range(10))          # prints "[0, 1, 4,..., 81]"
#######################################

# It is advised that a threaded pgogram should always arrange for a single thread to deal with
# any given object or subsystem that is external to the program (such as a file, a GUI, or a network connection).

# Whenever your threaded program must deal with some external object, devote a thread to such dealinds using a Queue object
# from which the external-interfacing thread gets work requests that other threads post.
# The external-interfacing thread can return results by putting them on one or more othe Queue objects.




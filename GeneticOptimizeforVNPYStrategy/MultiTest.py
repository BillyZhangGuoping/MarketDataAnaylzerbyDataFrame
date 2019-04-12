from multiprocessing import Pool
import time


def task(msg):
    print 'hello, %s' % msg
    time.sleep(1)
    return 'msg: %s' % msg



pool = Pool(processes=4)

results = []
msgs = [x for x in range(2)]

results = pool.map(task, msgs)

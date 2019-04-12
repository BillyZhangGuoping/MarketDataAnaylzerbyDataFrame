import multiprocessing


def func(y,x):
    return y * x
class someClass(object):
    def __init__(self,func):
        self.f = func

    def go(self):
        pool = multiprocessing.Pool(processes=4)
        lista = [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        print pool.map(self.f, lista, range(10))

if __name__== '__main__' :
    c = someClass(func)
    c.go()
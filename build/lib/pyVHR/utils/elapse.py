import time

def tic(verb=True):
    """ tic like Matlab function """

    global tic_toc_time
    tic_toc_time = time.time()
    if verb:
        print('start...')
    return tic_toc_time

def toc(verb=True):
    """ toc like Matlab function """

    global tic_toc_time
    T1 = time.time()-tic_toc_time
    if verb:
        print('elapsed = ' + str(T1))
    return T1

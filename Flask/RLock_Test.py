import threading
import time

g_Rlock = threading.RLock()

def t1():
    while True:
        g_Rlock.acquire()
        print('t1')
        g_Rlock.release()
        time.sleep(2)
    return

def t2():
    while True:
        g_Rlock.acquire()
        print('t2')
        g_Rlock.release()
        time.sleep(2)

    return


if __name__ == '__main__':
    tt1 = threading.Thread(target=t1)
    tt1.setDaemon(True)
    tt1.start()

    time.sleep(10)

    tt2 = threading.Thread(target=t2)
    tt2.setDaemon(True)
    tt2.start()

    time.sleep(3600)

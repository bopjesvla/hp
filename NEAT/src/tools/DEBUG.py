import time
import sys

prt_debug = False
log_debug = True

def dprt( obj, message ):
    if prt_debug:
        print( "%s@%s -> \"%s\"" % (str(obj), time.ctime(time.time()), str(message)) )

def dlog( obj, message ):
    if log_debug:
        with open('../log/log.txt','a') as f:
            f.write("%s || %s -> \"%s\"\n" % (time.ctime(time.time()), str(obj), str(message)))

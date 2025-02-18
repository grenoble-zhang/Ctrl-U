# Random a port between 30001 and 40000.
# test if it is occupied. if not, print it.
import socket
import random
port = random.randint(30001, 40000)
while True:
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        s.bind(('localhost', port))
        print(port)
        break
    except:
        port = random.randint(30001, 40000)
        continue

import sys, socket

'''
This is the summarization triggering command that the server
is listening to
'''
SUMMARIZATION_TRIGGER = "7XXASDHHCESADDFSGHHSD"

port = int(sys.argv[1])

s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.connect(("0.0.0.0", port))
s.send(SUMMARIZATION_TRIGGER.encode("utf-8"))

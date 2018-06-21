import sys, socket, argparse, json, time

'''
This is the summarization triggering command that the server
is listening to
'''
SUMMARIZATION_TRIGGER = "7XXASDHHCESADDFSGHHSD"

def run(args):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  
  tries = 0
  while(tries < args.maxWaitAttempts):
    try:
      s.connect(("0.0.0.0", args.port))
    except (RuntimeError,ConnectionRefusedError):
      tries += 1
      time.sleep(args.waitTime)
      if tries == args.maxWaitAttempts: 
        sys.exit(1)
        print("Failed to connect within the specified timeframe %d wait time x %d tries" % (args.waitTime,args.maxWaitAttempts))

  d = {"qExpansion": args.qExpansion, "qResults": args.qResults, "experiment": args.experiment, "dataStructure": args.dataStructure}
  s.send(json.dumps(d).encode("utf-8"))
  data = None
  while(data != SUMMARIZATION_TRIGGER):
   data = s.recv(1000000)
   data = str(data, "utf-8")
  print("Summarization request ended.")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--port", required=True, type=int)
  parser.add_argument("--qExpansion", required=True, type=str)
  parser.add_argument("--qResults", required=True, type=str)
  parser.add_argument("--experiment", required=True, type=str)
  parser.add_argument("--dataStructure", required=True, type=str)
  parser.add_argument("--waitTime", required=False, type=int, default=30)
  parser.add_argument("--maxWaitAttempts", required=False, type=int, default=60)
 
  args = parser.parse_args() 
  run(args)

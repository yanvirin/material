import sys, socket, argparse, json

'''
This is the summarization triggering command that the server
is listening to
'''

SUMMARIZATION_TRIGGER = "7XXASDHHCESADDFSGHHSD"

def run(port, qExpansion, qResults, experiment):
  s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
  s.connect(("0.0.0.0", port))
  d = {"qExpansion": qExpansion, "qResults": qResults, "experiment": experiment}
  s.send(json.dumps(d).encode("utf-8"))
  data = None
  while(data != SUMMARIZATION_TRIGGER):
   data = s.recv(1000000)
   data = str(data, "utf-8")
  print("Summarization results are ready!")

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("--port", required=True, type=int)
  parser.add_argument("--qExpansion", required=True, type=str)
  parser.add_argument("--qResults", required=True, type=str)
  parser.add_argument("--experiment", required=True, type=str)
 
  args = parser.parse_args() 
  run(port = args.port, qExpansion = args.qExpansion, qResults = args.qResults, experiment = args.experiment)

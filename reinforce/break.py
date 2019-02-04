import sys,json,os,string

input_file = sys.argv[1]
output_folder = sys.argv[2]

os.system("mkdir -p %s" % output_folder)

i = 0
for line in open(sys.argv[1]):
  d = json.loads(line)
  with open(output_folder + "/" + d["id"] + ".json", "w") as w:
    w.write(line.strip())
  i += 1
  if i % 100 == 0: print("Finished writing %d docs" %i)



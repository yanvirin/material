import os,sys,utils

input_dir = sys.argv[1]

for f in os.listdir(input_dir):
  text = utils.fileaslist("%s/%s" % (input_dir,f))
  clean = []
  for sen in text:
    if len(sen.split(" ")) >= 4:
      clean.append(sen)
  if len(text) > len(clean): print "%d => %d" % (len(text), len(clean))
  utils.write2file("\n".join(clean),"%s/%s" % (input_dir, f))

import sys,os,re

extractor = re.compile("\>(.*)\<")

path = sys.argv[1]
path2 = sys.argv[2]

for d in os.listdir(path):
  file_path = "%s/%s/%s" % (path,d,100)
  if os.path.exists(file_path):
    with open(file_path) as r:
      raw_text = r.read().replace("\n", " ")
      text = extractor.findall(raw_text)[0]
    file_path2 = "%s/%s.%s.100" % (path2,d[:-1].upper(),d[-1].upper())
    with open(file_path2, "w") as w:
      w.write(text)

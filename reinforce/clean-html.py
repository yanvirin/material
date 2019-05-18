import os,sys,re
import html2text
from nltk.tokenize import sent_tokenize

p = re.compile("TEXT>(.*)</TEXT")

input_path = sys.argv[1]
output_path = sys.argv[2]

os.system("mkdir -p %s"%output_path)

for d in os.listdir(input_path):
  for f in os.listdir("%s/%s"%(input_path,d)):
    print("working on %s/%s" % (d, f))
    with open("%s/%s/%s"%(input_path,d,f)) as r:
      text = html2text.html2text(p.findall(r.read().replace("\n"," "))[0])
    write_dir = "%s/%s"%(output_path,d)
    os.system("mkdir -p %s"%write_dir)
    with open("%s/%s"%(write_dir,f),"w") as w:
      for sen in sent_tokenize(text.replace("\n"," ")):
        if len(sen) > 4 and len(sen.split(" ")) < 400:
          w.write(sen+"\n")

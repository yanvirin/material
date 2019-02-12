import os,sys,re
from nltk.tokenize import sent_tokenize

re.compile("


input_path = sys.argv[1]
output_path = sys.argv[2]

os.system("mkdir -p %s"%output_path)

for d in os.listdir(input_path):
  for f in os.listdir("%s/%s"%(input_path,d)):
    with open("%s/%s/%s"%(input_path,d,f)) as r:
      text = html2text.html2text(r.read())
    write_dir = "%s/%s"%(output_path,d)
    os.system("mkdir -p %s"%write_dir)
    with open("%s/%s"%(write_dir,f),"w") as w:
      for sen in sent_tokenize(text.replace("\n"," ")):
        if len(sen) > 4 and len(sen.split(" ")) < 40:
          w.write(sen+"\n")

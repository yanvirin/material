import sys
import re
import string
from select_urls import file_id
from os import path

url_regex = "url=(.+)"
url_pattern = re.compile(url_regex)

min_lines = int(sys.argv[1])

i = 1
n = 0
YES=0
NO=0
found = set()

for line in open(sys.argv[2]):
  line = line.strip()
  line = "".join(filter(lambda x: x in string.printable, line))
  url_m = url_pattern.search(line)
  article_id = str(i) + " "
  
  if article_id in line:
    n += 1
    i += 1
    title_tokens = set(line[len(str(i))+1:].split(" "))
  
  if url_m:
    url = url_m.group(1).split("|")[0]
    fid = file_id(url)
    if path.exists("sources/txt/%s.txt" % fid):
      with open("sources/txt/%s.txt" % fid, "r") as f:
        text = f.read()
        tokens = set(text.replace("\n", " ").split(" "))
        if len(filter(lambda x: len(x)>0, text.split("\n"))) > min_lines and tokens.intersection(title_tokens) > len(title_tokens)/1.2:
          if n not in found:
            print text
            print "YES " + str(YES) + " NO " + str(NO)
            print "title ===> " + " ".join(title_tokens)
            YESNO = raw_input('VALID======> ')
            if YESNO == "y": YES+=1
            if YESNO == "n": NO+=1
          found.add(n)

print n
print len(found)
  

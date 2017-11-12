import sys
import re
from os import system
from os import path
from goose import Goose
import codecs
from utils import file_id

def save_article_html(url):
  fid = file_id(url)
  url = url.replace("'","")
  system("wget -O sources/html/%s.html --tries=2 --wait=1 '%s' >> sources/logs/%s.log 2>> sources/logs/%s.log" % (fid, url, fid, fid))

if __name__ == "__main__":
  # make sure we have all needed folders
  system("mkdir -p sources")
  system("mkdir -p sources/html")
  system("mkdir -p sources/txt")
  system("mkdir -p sources/logs")

  url_regex = "url=(.+)"
  pattern = re.compile(url_regex)

  g = Goose()
  for line in sys.stdin:
    m = pattern.search(line)
    try:
     if m:
      url = m.group(1).split("|")[0]
      fid = file_id(url)
    
      # don't do work for already existing material
      if not path.exists("sources/html/%s.html" % fid):
        save_article_html(url)
        with open("sources/html/%s.html" % fid, "r") as f: 
          html = f.read()
        if len(html) > 0:
          a = g.extract(raw_html=html)
          with open("sources/txt/%s.txt" % fid, "w") as f:
            wr = codecs.getwriter('utf8')
            wr(f).write("%s\n\n%s" % (a.title, a.cleaned_text))
    except Exception as e:
      print "Error was encountered: %s" % e

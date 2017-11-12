import sys, re, os, string
from utils import file_id
from os import system

url_regex = "url=(.+)"
url_pattern = re.compile(url_regex)

entity_regexes = [r"\[\[[^\]]+?\|([^\]]+?)\|[^\]]+?\]\]",r"\{\{[^\}]+?\|([^\}]+?)\|[^\}]+?\}\}",r"\[\[[^\]]+?\|([^\]]+?)\]\]",r"\{\{[^\}]+?\|([^\}]+?)\}\}",r"\[\[([^\]]+?)\]\]","\{\{([^\}]+?)\}\}"," (\.)$"]
entity_patterns = [re.compile(r1) for r1 in entity_regexes]
end_regexes = [r"\{\{haveyoursay\}\}",r"== Sources ==",r"==Sources==",r"==Source==",r"== Source ==","== Related news ==","url="]
end_patterns = [re.compile(r2) for r2 in end_regexes]

i = 1
n = 0
content = []
record_content = False

sources_folder = sys.argv[1]
out_folder = sys.argv[2]

system("mkdir -p %s" % out_folder)

for line in sys.stdin:
  line = line.strip()
  line = "".join(filter(lambda x: x in string.printable, line))
  url_m = url_pattern.search(line)
  
  article_id = str(i) + " "
  
  if article_id in line:
    n += 1
    i += 1
    content = [line[len(article_id):]]
    system("mkdir -p %s/%d/sources" % (out_folder, n))
    record_content = True
  else:
    if record_content:
     for p in end_patterns:
      if p.search(line):
        with open("%s/%d/content.txt" % (out_folder, n), "w") as w: w.write("\n".join(content))
        record_content = False
        break

    if record_content:
      for p in entity_patterns:
        line = re.sub(p, "\g<1> ", line)
      content.append(line)
  
  if url_m:
    url = url_m.group(1).split("|")[0]
    fid = file_id(url)
    source_file = "%s/txt/%s.txt" % (sources_folder, fid)
    out_file = "%s/%d/sources/%s.txt" % (out_folder, n, fid)
    if os.path.exists(source_file):
      code = system("cp %s %s" % (source_file, out_file))
      if os.WEXITSTATUS(code) != 0: raise Exception("Failed to copy!")

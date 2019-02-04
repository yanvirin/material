import sys, random, os

important_words = ["car","machine","vechicle","ride","trip","drive","move","speed","travel","go","compete","license","drunk","violation",
                   "ticket","law","road","drink","sign","work","commute","traffic","bridge","toll","ignition","engine","power","tork","hourse",
                   "mpg","efficiency","electric","gas","diesel","seat","comfort","heavy","went","die","accident","incident","police","fire","explode",
                   "tesla","toyota","subaru","honda","mazda","kia","leasing","buy","new"]
other_words = ["air","water","computer","friends","sun","keyboard","legs","body","monitor","colors","blue","white","people","nothing","food","garbage","line","circle",
               "see","deny","love","percent","item","store","dollar","money","valley","cost","line","divide","math","number","do","did","several","stain","house",
               "today","weather","thanks","say","agent","message","register","listen","ring","sevage","different","lie","custom","worker","teacher","proffesor",
               "student","name","picture","book","letter","write","form","corner","table"]


probs = [0.05, 0.1, 0.2, 0.4, 0.65, 0.85, 0.95, 0.99, 1.0] 

sentence_length = 10
num_sentences = 10

def choose_n(n, l):
  return [random.choice(l) for x in range(n)]

def choose_p(probs):
  x = random.random()
  for i,p in enumerate(probs):
    if x <= p: return (i+1)/10


# generate documents
docs_path = sys.argv[1]
num_of_docs = int(sys.argv[2])
random.seed(num_of_docs)

os.system("mkdir -p %s" % docs_path)

for d in range(num_of_docs):
  summary = " ".join(choose_n(sentence_length,important_words))
  content = []
  for s in range(num_sentences):
    portion = choose_p(probs)
    important = int(portion*sentence_length)
    other = sentence_length-important
    sentence = " ".join(choose_n(important,important_words) + choose_n(other,other_words))
    content.append(sentence)
    if random.random() <= portion:
      tokens = sentence.split(" ")
      content.append(" ".join([random.choice(tokens) for x in range(int(len(tokens)*portion))]))
  with open("%s/%s.data" % (docs_path,str(d)), "w") as w:
    w.write("\n\n\n\n"+summary+"\n\n\n"+"\n".join(content))

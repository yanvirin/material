import random,sys,os

SEED = 354886748

inputs_dir = sys.argv[1]
labels_dir = sys.argv[2]
train = float(sys.argv[3])
dev = float(sys.argv[4])
test = float(sys.argv[5])

assert(train+dev+test==1.0)

random.seed(SEED)

TRAIN = "train"
DEV = "dev"
TEST = "test"

os.system("mkdir -p %s/%s" % (inputs_dir,TRAIN))
os.system("mkdir -p %s/%s" % (inputs_dir,DEV))
os.system("mkdir -p %s/%s" % (inputs_dir,TEST))
os.system("mkdir -p %s/%s" % (labels_dir,TRAIN))
os.system("mkdir -p %s/%s" % (labels_dir,DEV))
os.system("mkdir -p %s/%s" % (labels_dir,TEST))

for f in os.listdir(inputs_dir):
  if not os.path.isfile("%s/%s" % (inputs_dir,f)): continue
  folder = None
  number = random.random()
  if number < train:
    folder = TRAIN
  elif number < train+dev:
    folder = DEV
  else:
    folder = TEST

  assert(folder)
  
  os.system("mv %s/%s %s/%s/%s" % (inputs_dir,f,inputs_dir,folder,f))
  os.system("mv %s/%s %s/%s/%s" % (labels_dir,f,labels_dir,folder,f))
 

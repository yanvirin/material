import zipfile
import collections
import numpy as np

import math
import random
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torch.nn.functional as Func
from torch.optim.lr_scheduler import StepLR
import time

from inputdata import Options, scorefunction
from model import skipgram

class word2vec:
  def __init__(self, inputfile, embeddingsfile, reg_lambda=0.000001, embedding_dim=300, epoch_num=5, batch_size=16, windows_size=5, neg_sample_num=10):
    self.op = Options(inputfile, embeddingsfile, embedding_dim)
    self.windows_size = windows_size
    self.vocabulary_size = self.op.vocabulary_size
    self.batch_size = batch_size
    self.epoch_num = epoch_num
    self.neg_sample_num = neg_sample_num
    self.reg_lambda = reg_lambda

  def train(self):
    model = skipgram(self.vocabulary_size, self.op.embeddings.shape[1], reg=self.reg_lambda)
    model.init_emd(self.op.embeddings)
 
    if torch.cuda.is_available():
      model.cuda()
    optimizer = optim.SGD(model.parameters(),lr=0.2)
    for epoch in range(self.epoch_num):
      start = time.time()     
      self.op.process = True
      batch_num = 0
      batch_new = 0

      while self.op.process:
        pos_u, pos_v, neg_v = self.op.generate_batch(self.windows_size, self.batch_size, self.neg_sample_num)

        pos_u = Variable(torch.LongTensor(pos_u))
        pos_v = Variable(torch.LongTensor(pos_v))
        neg_v = Variable(torch.LongTensor(neg_v))


        if torch.cuda.is_available():
          pos_u = pos_u.cuda()
          pos_v = pos_v.cuda()
          neg_v = neg_v.cuda()

        optimizer.zero_grad()
        loss = model(pos_u, pos_v, neg_v,self.batch_size)

        loss.backward()
   
        optimizer.step()

        if batch_num%30000 == 0:
          torch.save(model.state_dict(), './tmp/skipgram.epoch{}.batch{}'.format(epoch,batch_num))

        if batch_num%2000 == 0:
          end = time.time()
          word_embeddings = model.input_embeddings()
          sp1, sp2 = scorefunction(word_embeddings)     
          print 'epoch,batch=%2d %5d: sp=%1.3f %1.3f  pair/sec = %4.2f loss=%4.3f' %(epoch, batch_num, sp1, sp2, (batch_num-batch_new)*self.batch_size/(end-start),loss.data[0])
          batch_new = batch_num
          start = time.time()
        batch_num = batch_num + 1 
      print()
    print("Optimization Finished!")
    model.save_embedding("embeddings.txt", lambda x: self.op.vocab_words[x])

  
if __name__ == '__main__':
  wc = word2vec(sys.argv[1], sys.argv[2], epoch_num=10, reg_lambda=float(sys.argv[3]))
  wc.train()

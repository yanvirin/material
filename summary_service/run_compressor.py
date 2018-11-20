# Jessica Ouyang
# run_compressor.py


import _pickle as cPickle
from itertools import groupby
from operator import itemgetter
from os import listdir
import os.path
from random import shuffle
import sys
import time

import numpy as np

import torch
from torch import optim

import compressor as copy_net


use_cuda = torch.cuda.is_available()

START, END, UNK = 50000, 50001, 50002


def load_src(src_list, embedding_lookup, constraints_list, b):
    num_src = len(src_list)
    max_src_len = max([len(sent) for sent in src_list]) + 1
    
    unscrambler = {}
    length_sorted = sorted(src_list, key=lambda x: len(x), reverse=True)
    for i in range(num_src):
        unscrambler[length_sorted.index(src_list[i])] = i
        
    extended_lookup, extended_counter = {}, 0
    src_map_i, src_map_j, src_map_k, src_map_v = [], [], [], []
        
    output = [0] * num_src
    for i in range(num_src):
        src_sent = src_list[i]
        src_sent.append('<end>')
        src_len = len(src_sent)
        
        src_input = np.full(max_src_len, END, dtype=np.int64)
        for j in range(src_len):                
            word = src_sent[j]

            src_map_i.append(i)
            src_map_j.append(j)
            src_map_v.append(1)

            if word in embedding_lookup:
                src_input[j] = embedding_lookup[word]
            else:                    
                src_input[j] = UNK
                    
            if not word in extended_lookup:
                extended_lookup[word] = extended_counter
                extended_counter += 1                        
            src_map_k.append(extended_lookup[word])

        word_constraints, phrasal_constraints = [], []
        for constraint in constraints_list:
            if constraint[0] in src_sent:
                if len(constraint) == 1:
                    word_constraints.append(extended_lookup[constraint[0]])
                else:
                    constraint_found = True
                    index = src_sent.index(constraint[0])
                    for k in range(1, len(constraint)):
                        if src_sent[index + k] != constraint[k]:
                            constraint_found = False
                            break
                    if constraint_found:                        
                        phrasal_constraints.append([extended_lookup[word] for word in constraint])
                    
        output[i] = (src_input, src_len, word_constraints, phrasal_constraints)
    output = sorted(output, key=itemgetter(1), reverse=True)

    src_inputs, src_lens = np.zeros((num_src, max_src_len), dtype=np.int64), np.zeros(num_src, dtype=np.int64)
    src_map = torch.sparse.FloatTensor(num_src, max_src_len, extended_counter)
    word_constraints, phrasal_constraints = [0] * num_src, [0] * num_src
        
    for i in range(num_src):
        src_input, src_len, wcon, pcon = output[i]
        src_inputs[i, :], src_lens[i] = src_input, src_len
        word_constraints[i], phrasal_constraints[i] = wcon, pcon
            
        src_map_i, src_map_j, src_map_k = torch.LongTensor(src_map_i).view(1, -1), torch.LongTensor(src_map_j).view(1, -1), torch.LongTensor(src_map_k).view(1, -1)
        src_map_v = torch.FloatTensor(src_map_v)
    src_map = torch.sparse.FloatTensor(
        torch.cat([src_map_i, src_map_j, src_map_k], 0),
        src_map_v,
        torch.Size([num_src, max_src_len, extended_counter]))

    reverse_extended_lookup = {v:k for k,v in extended_lookup.items()}
    return (torch.from_numpy(src_inputs), torch.from_numpy(src_lens), src_map, reverse_extended_lookup, word_constraints, phrasal_constraints, unscrambler)


def compress(compressor, src, constraints, b=5, k=4):
    encoder, decoder, embedding_lookup = compressor
    
    src, src_len, src_map, reverse_extended_lookup, word_constraints, phrasal_constraints, unscrambler = load_src(src, embedding_lookup, constraints, b)
    if use_cuda:
        src, src_len, src_map = src.cuda(), src_len.cuda(), src_map.cuda()

    batch_size = src.size()[0]
    predictions = copy_net.predict(src, src_len, src_map, encoder, decoder, word_constraints, phrasal_constraints, k=k)
    del src, src_len, src_map

    all_output = [0] * batch_size
    for i in range(batch_size):
        output = [reverse_extended_lookup[word.item()] for word in predictions[i]]
        if '<end>' in output:
            output = output[:output.index('<end>')]

        # truncate at three of the same token in a row (degenerate)
        the_group = None
        for _, g in groupby(output):
            group = list(g)
            if len(group) > 2:
                the_group = group
                break
            
        output = ' '.join(output)
        if the_group != None:
            output = output[:output.index(' '.join(the_group))]

        # remove two of the same token in a row
        output = output.split()
        cleaned_output = [output[0]]
        for j in range(1, len(output)):
            if output[j] != output[j-1]:
                cleaned_output.append(output[j])

        all_output[unscrambler[i]] = cleaned_output
    return all_output

            
def load_compressor(model_fname, embedding_lookup):
    encoder = copy_net.Encoder(512, 300, 50003)
    decoder = copy_net.Decoder(512, 300, 50003)
    if use_cuda:
        encoder, decoder = encoder.cuda(), decoder.cuda()

    encoder.load_state_dict(torch.load(model_fname + '.encoder'))
    decoder.load_state_dict(torch.load(model_fname + '.decoder'))

    with open(embedding_lookup, 'rb') as inf:
        lookup = cPickle.load(inf, encoding='bytes')
    return (encoder, decoder, lookup)
    

#def main(args):
#    model_fname = args[0]
#    embedding_lookup = args[1]

#    src = [['the', 'apple', 'pie', 'will', 'sell', 'for', '10', 'bobs', '.'],
#               ['the', 'apple', 'strudel', 'will', 'sell', 'for', '8', 'bobs', 'and', 'the', 'bushel', 'of', 'oranges', 'will', 'sell', 'for', '11', 'bobs', '.']]

#    constraints = [['apple', 'pie'],
#                       ['strudel'],
#                       ['apple'],
#                       ['pie']]

#    compressor = load_compressor(model_fname, embedding_lookup)
#    output = compress(compressor, src, constraints)
#    for sent in output:
#        print ' '.join(sent)
    
#if __name__ == '__main__':
#    sys.exit(main(sys.argv[1:]))

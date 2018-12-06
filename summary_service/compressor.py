# Jessica Ouyang
# compressor.py
# aka copy_net.py


from math import sqrt
from operator import itemgetter
import random

import torch
import torch.nn as nn
import torch.nn.functional as F

use_cuda = torch.cuda.is_available()

START, END, UNK = 50000, 50001, 50002
epsilon = 1e-06


class Encoder(nn.Module):
    def __init__(self, hidden_size, embed_size, vocab_size, n_layers=1, dropout=0):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.embed_size = embed_size
        
        self.embedding = nn.Embedding(vocab_size, embed_size)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, n_layers, dropout=dropout, bidirectional=True, batch_first=True)

    # input_seqs: shape (batch size, max sequence length), by decreasing length
    # input_lengths: list of sequence lengths
    def forward(self, input_seqs, input_lengths, state):
        output, state = self.lstm(
            nn.utils.rnn.pack_padded_sequence(
                self.embedding(input_seqs),
                input_lengths, batch_first=True),
            state)
        del input_seqs, input_lengths
        output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        return (output, state)

    def init_state(self, batch_size):
        state = (torch.randn(2, batch_size, self.hidden_size),
                  torch.randn(2, batch_size, self.hidden_size))
        if use_cuda:
            state = (state[0].cuda(), state[1].cuda())
        return state

        
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()
        self.hidden_size = hidden_size

        self.attn = nn.Linear(hidden_size * 4, hidden_size)
        self.v = nn.Parameter(torch.rand(hidden_size))
        self.v.data.normal_(mean=0, std=(1. / sqrt(hidden_size)))

    # decoder_state: shape (1, batch size, hidden size)
    # encoder_outputs: shape (batch size, max length, 2 * hidden size)
    # coverage: shape (batch size, max length)
    def forward(self, decoder_state, encoder_outputs, coverage):
        weights = self.score(
            decoder_state.repeat(encoder_outputs.size()[1], 1, 1).transpose(0, 1),
            encoder_outputs,
            coverage.unsqueeze(2).repeat(1, 1, self.hidden_size))
        del decoder_state, encoder_outputs

        weights = F.softmax(weights, 1)

        coverage = coverage + weights
        return (weights, coverage)
        
    # decoder_state: shape (1, batch size, hidden size)
    # encoder_outputs: shape (batch size, max length, 2 * hidden size)
    # coverage: shape (batch size, max length)
    # returns: shape (batch size, max length)
    def score(self, decoder_state, encoder_outputs, coverage):
        return torch.bmm(
            self.v.repeat(encoder_outputs.data.shape[0], 1).unsqueeze(1),
            F.tanh(self.attn(
                torch.cat([decoder_state, encoder_outputs, coverage], 2))).transpose(2, 1)).squeeze(1)
            
            
class Decoder(nn.Module):
    def __init__(self, hidden_size, embed_size, vocab_size, n_layers=1, dropout=0):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.n_layers = n_layers

        self.embedding = nn.Embedding(vocab_size, embed_size)

        self.attn = Attention(hidden_size)
        self.lstm = nn.LSTM(2 * hidden_size + embed_size, hidden_size, n_layers, dropout=dropout, batch_first=True)
        
    # prev_word: shape (batch size, 1)
    # src_map: shape (batch size, max src len, extended vocab size)
    def forward(self, prev_word, state, encoder_outputs, coverage, src_map):
        attn_weights, coverage = self.attn(state[0], encoder_outputs, coverage)

        _, state = self.lstm(
            torch.cat([self.embedding(prev_word),
                           attn_weights.unsqueeze(1).bmm(encoder_outputs)],
                          2),
            state)       
        del prev_word, encoder_outputs
        
        p_copy = torch.bmm(attn_weights.unsqueeze(2).transpose(1, 2),
                               src_map.to_dense()).squeeze(1) + epsilon
        del src_map
        return (p_copy, attn_weights, coverage, state)


# constraints: list of lists of word ids (matching src_map), one per batch item
# phrasal_constraints: same but lists of tuples
def predict(src, src_lens, src_map, encoder, decoder, constraints=None, phrasal_constraints=None, max_length=100, k=4):
    batch_size, max_src_len = src.size()

    if not constraints:
        constraints = [[]] * batch_size
    if not phrasal_constraints:
        phrasal_constraints = [[]] * batch_size

    C = [len(constraints[bi]) +
             len([p for pcon in phrasal_constraints[bi] for p in pcon])
             for bi in range(batch_size)]
    bank_size = [k/(c+1) for c in C]    
    
    encoder_state = encoder.init_state(batch_size)
    encoder_outputs, encoder_state = encoder(src, src_lens, encoder_state)
    del src, src_lens
    
    prev_token = torch.LongTensor([[START]]).repeat(batch_size, 1)
    coverage = torch.zeros(max_src_len).repeat(batch_size, 1)
    if use_cuda:
        prev_token, coverage = prev_token.cuda(), coverage.cuda()
        
    decoder_state = (encoder_state[0].mean(0, keepdim=True), encoder_state[1].mean(0, keepdim=True))
    
    p_copy, _, coverage, decoder_state = decoder(prev_token, decoder_state, encoder_outputs, coverage, src_map)    
    extended_vocab_size = p_copy.size()[1]
    
    beam = []
    for bi in range(batch_size):
        beam.append([])

        scores, vis = torch.topk(p_copy[bi,:], k)
        scores, vis = scores.cpu(), vis.cpu()        
        for vi in range(vis.size()[0]):
            token_id = vis[vi].item()

            filled = [0] * len(constraints[bi])
            if token_id in constraints[bi]:
                filled[constraints[bi].index(token_id)] = 1

            num_phrasal = len(phrasal_constraints[bi])
            filled_phrasal = [0] * num_phrasal
            for pi in range(num_phrasal):
                if phrasal_constraints[bi][pi][0] == token_id:
                    filled_phrasal[pi] = 1                    
                
            beam[bi].append((
                (decoder_state[0][:,bi,:].unsqueeze(1),
                     decoder_state[1][:,bi,:].unsqueeze(1)),
                [torch.LongTensor([[token_id]])],
                scores[vi],
                coverage[bi,:].unsqueeze(0),
                filled, filled_phrasal))

    for di in range(1, max_length):
        all_candidates = []
        for bi in range(batch_size):
            all_candidates.append([])

        for ki in range(k):
            batch_prev_state = [0] * batch_size
            batch_prev_token = [0] * batch_size
            batch_prev_score = [0] * batch_size
            batch_prev_coverage = [0] * batch_size
            batch_filled = [0] * batch_size
            batch_phrasal = [0] * batch_size

            all_prev_tokens = [0] * batch_size
            for bi in range(batch_size):                
                prev_state, prev_tokens, prev_score, prev_coverage, prev_filled, prev_phrasal = beam[bi][ki]
                batch_prev_state[bi] = prev_state # tuple of size 2
                batch_prev_token[bi] = torch.clamp(prev_tokens[-1], 0, UNK)
                if use_cuda:
                    batch_prev_token[bi] = batch_prev_token[bi].cuda()
                batch_prev_score[bi] = prev_score
                batch_prev_coverage[bi] = prev_coverage
                batch_filled[bi] = prev_filled
                batch_phrasal[bi] = prev_phrasal

                all_prev_tokens[bi] = prev_tokens
                del prev_state, prev_coverage
                
            p_copy, _, coverage, decoder_state = decoder(
                torch.cat(batch_prev_token, 0),
                (torch.cat([pair[0] for pair in batch_prev_state], 1),
                     torch.cat([pair[1] for pair in batch_prev_state], 1)),
                encoder_outputs,
                torch.cat(batch_prev_coverage, 0),
                src_map)
            del batch_prev_state, batch_prev_token, batch_prev_coverage

            for bi in range(batch_size):
                scores, vis = torch.topk(p_copy[bi,:], k)
                scores, vis = scores.cpu(), vis.cpu()
                
                prev_filled, prev_phrasal = batch_filled[bi], batch_phrasal[bi]                
                num_phrasal = len(phrasal_constraints[bi])

                # k best
                for vi in range(vis.size()[0]):
                    token_id = vis[vi].item()
                    token = torch.LongTensor([[token_id]])
                    new_tokens = all_prev_tokens[bi] + [token]

                    new_filled = list(prev_filled)
                    if token_id in constraints[bi]:
                        new_filled[constraints[bi].index(token_id)] = 1

                    new_phrasal = list(prev_phrasal)
                    for pi in range(num_phrasal):
                        pcount = prev_phrasal[pi]
                        # constraint begun but not yet finished
                        if pcount > 0 and pcount < len(phrasal_constraints[bi][pi]):
                            if token_id == phrasal_constraints[bi][pi][pcount]: # finishing it out
                                new_phrasal[pi] = pcount + 1
                            else: # not finishing; roll back
                                new_phrasal[pi] = pcount - 1

                    all_candidates[bi].append((
                        (decoder_state[0][:,bi,:].unsqueeze(1),
                             decoder_state[1][:,bi,:].unsqueeze(1)),
                        new_tokens, batch_prev_score[bi] * scores[vi],
                        coverage[bi,:].unsqueeze(0),
                        new_filled, new_phrasal))

                # progress constraints
                for token_id in constraints[bi]:
                    if not token_id in vis:
                        token = torch.LongTensor([[token_id]])
                        new_tokens = all_prev_tokens[bi] + [token]

                        new_filled = list(prev_filled)
                        new_filled[constraints[bi].index(token_id)] = 1

                        new_phrasal = list(prev_phrasal)
                        for pi in range(num_phrasal):
                            progress = prev_phrasal[pi]
                            if progress > 0 and progress < len(phrasal_constraints[bi][pi]): 
                                if token_id == phrasal_constraints[bi][pi][progress]:
                                    new_phrasal[pi] = progress + 1
                                else:
                                    new_phrasal[pi] = progress - 1

                        all_candidates[bi].append((
                            (decoder_state[0][:,bi,:].unsqueeze(1),
                                decoder_state[1][:,bi,:].unsqueeze(1)),
                            new_tokens, batch_prev_score[bi] * p_copy[bi,token_id].cpu(),
                            coverage[bi,:].unsqueeze(0),
                            new_filled, new_phrasal))

                # progress phrasal constraints
                for pi in range(num_phrasal):
                    if prev_phrasal[pi] >= len(phrasal_constraints[bi][pi]): # finished
                        continue

                    token_id = phrasal_constraints[bi][pi][prev_phrasal[pi]]
                    if not token_id in vis:
                        token = torch.LongTensor([[token_id]])
                        new_tokens = all_prev_tokens[bi] + [token]

                        new_filled = list(prev_filled)
                        if token_id in constraints[bi]:
                            new_filled[constraints[bi].index(token_id)] = 1                        

                        new_phrasal = list(prev_phrasal)
                        new_phrasal[pi] = new_phrasal[pi] + 1

                        all_candidates[bi].append((
                            (decoder_state[0][:,bi,:].unsqueeze(1),
                                decoder_state[1][:,bi,:].unsqueeze(1)),
                            new_tokens, batch_prev_score[bi] * p_copy[bi,token_id].cpu(),
                            coverage[bi,:].unsqueeze(0),
                            new_filled, new_phrasal))
            del prev_tokens, batch_prev_score

        for bi in range(batch_size):
            new_beam = []
            banks = [bank_size[bi]] * (C[bi] + 1)
            banks[-1] = banks[-1] + k - sum(banks)

            bank_candidates = [[candidate for candidate in all_candidates[bi]
                                    if sum(candidate[4]) + sum(candidate[5]) == c]
                                   for c in range(C[bi] + 1)]

            for c in range(C[bi], -1, -1):
                bank_candidates[c].sort(key=itemgetter(2), reverse=True)
                num_extra_banks = banks[c] - len(bank_candidates[c])
                if num_extra_banks > 0: # more banks than candidates
                    new_beam.extend(bank_candidates[c])
                    if c > 0:
                        banks[c-1] = banks[c-1] + num_extra_banks
                else:
                    new_beam.extend(bank_candidates[c][:int(banks[c])])

            beam[bi] = new_beam
            del new_beam
    return tuple(batch_beam[0][1] for batch_beam in beam) # tokens lists

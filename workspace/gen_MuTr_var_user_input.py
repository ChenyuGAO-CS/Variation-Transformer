#
# Generate MIDI piano pieces with Music Transformer.
#
# Author: Chenyu Gao
#
#

import os
import torch
import json
import math
import argparse
import time

from encoder_var_gen import *
from transformer import Transformer

from torch.distributions.categorical import Categorical

END_TOKEN = Event(event_type='control', value=2).to_int()
BAR_TOKEN = Event(event_type='control', value=1).to_int()

def filter_top_p(y_hat, p, filter_value=-float("Inf")):
    sorted_logits, sorted_indices = torch.sort(y_hat, descending=True)
    cumulative_probs = sorted_logits.softmax(dim=-1).cumsum(dim=-1)

    # Remove tokens with cumulative top_p above the threshold (token with 0 are kept)
    sorted_indices_to_remove = cumulative_probs > p

    # Shift the indices to the right to keep also the first token above the threshold
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    # scatter sorted tensors to original indexing
    indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
    y_hat = y_hat.masked_fill(indices_to_remove, filter_value)

    return y_hat

def filter_top_k(y_hat, k, filter_value=-float("Inf")):
    # Remove all tokens with a probability less than the last token of the top-k
    indices_to_remove = y_hat < torch.topk(y_hat, k)[0][..., -1, None]
    y_hat = y_hat.masked_fill(indices_to_remove, filter_value)

    return y_hat

def filter_index(y_hat, index, filter_value=-float("Inf")):
    y_hat[:,index] = filter_value
    return y_hat

def filter_repetition(previous_tokens, scores, penalty=1.0001):
    score = torch.gather(scores, 1, previous_tokens)

    # if score < 0 then repetition penalty has to be multiplied to reduce the previous token probability
    score = torch.where(score < 0, score * penalty, score / penalty)

    scores.scatter_(1, previous_tokens, score)
    return scores

def sample_tokens(y_hat, num_samples=1):
    # Sample from filtered categorical distribution
    probs = torch.softmax(y_hat, dim=1)
    random_idx = torch.multinomial(probs, num_samples)
    return random_idx

def is_terminal(state, n_bars, seq_len):
    return torch.sum(state == BAR_TOKEN) >= n_bars or len(state) >= seq_len or state[-1] == END_TOKEN

def generate(model, prime, n_bars, seq_len, k=0, p=0, temperature=1.0):
    # Generate new tokens
    generated = torch.tensor(prime).unsqueeze(dim=0).to(device)
    
    while not is_terminal(generated.squeeze(), n_bars, seq_len):
        # import pdb
        # pdb.set_trace()
        # print("generated", generated)
        y_i = model(generated)[:,-1,:]

        # Filter out end token
        y_i = filter_index(y_i, END_TOKEN)

        if k > 0:
            y_i = filter_top_k(y_i, k)
        if p > 0 and p < 1.0:
            y_i = filter_top_p(y_i, p)

        token = sample_tokens(y_i)
        generated = torch.cat((generated, token), dim=1)

    return [int(token) for token in generated.squeeze(0)]

if __name__ == '__main__':
    # Parse arguments
    parser = argparse.ArgumentParser(description='generate.py')
    parser.add_argument('--lm', type=str, required=True, help="Path to load model from.")
    parser.add_argument('--emotion', type=int, default=0, help="Target emotion.")
    parser.add_argument('--seq_len', type=int, required=True, help="Max sequence to process.")
    parser.add_argument('--n_bars', type=int, default=4, help="Num bars to generate.")
    parser.add_argument('--k', type=int, default=0, help="Number k of elements to consider while sampling.")
    parser.add_argument('--p', type=float, default=1.0, help="Probability p to consider while sampling.")
    parser.add_argument('--t', type=float, default=1.0, help="Sampling temperature.")
    parser.add_argument('--n_layers', type=int, default=8, help="Number of transformer layers.")
    parser.add_argument('--d_model', type=int, default=512, help="Dimension of the query matrix.")
    parser.add_argument('--n_heads', type=int, default=8, help="Number of attention heads.")
    parser.add_argument('--device', type=str, default=None, help="Torch device.")
    parser.add_argument('--prime', type=str, required=False, help="Prime sequence.")
    parser.add_argument('--save_to', type=str, required=True, help="Directory to save the generated samples.")
    parser.add_argument('--input', type=str, required=True, help="Path of the input MIDI file.")
    parser.add_argument('--seed', type=int, required=None, help="Set a random seed.")
    opt = parser.parse_args()

    # Set up torch device
    device = opt.device
    if not device:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    vocab_size = VOCAB_SIZE

    if opt.seed:
        torch.manual_seed(opt.seed)

    # Build linear transformer
    model = Transformer(vocab=vocab_size, 
                        n_layer=opt.n_layers,
                        n_head=opt.n_heads,
                        d_model=opt.d_model).to(device)

    # Load model
    model.load_state_dict(torch.load(opt.lm, map_location=device)["model_state"])
    model.eval()

    # Define prime sequence from a user provided MIDI file. 
    MIDI_PATH = opt.input
    prime1 = encode_user_input_midi(MIDI_PATH)
    prime1.pop() # Pop the final token as it is "STOP".
    # prime1 += [0 for i in range(512-len(prime1))]
    len_prime1 = len(prime1)

    prime2 = [Event(event_type='control', value=0).to_int(), 
            Event(event_type='emotion', value=opt.emotion).to_int(),
            Event(event_type='beat', value=0).to_int()]
    
    prime = prime1 + [520] + prime2

    # Generate piece
    T1 = time.time()
    piece = generate(model, prime, n_bars=opt.n_bars, seq_len=opt.seq_len, k=opt.k, p=opt.p, temperature=opt.t)
    # decode_midi(piece, opt.save_to)
    decode_midi(piece[len_prime1+1:], opt.save_to)
    T2 = time.time()
    # print(piece)
    print('Time for inference: %s minutes' % ((T2 - T1)/(60)))
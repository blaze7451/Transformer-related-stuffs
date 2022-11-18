import argparse

import numpy as np
import torch
import torch.nn.functional as F

from dataset import TextDataset
from model import Net

def parse_args():
    parser = argparse.ArgumentParser(description='Generate text')
    parser.add_argument('--corpus', type=str, default="C:\\Python\\Pytorch\\Transformer related\\Chinese text generation\\Output.txt",
                        help='training corpus file')
    parser.add_argument('--output_model', type=str, default="C:\\Python\\Pytorch\\Transformer related\\Chinese text generation\\output_model.pt",
                        help='output model file')
    parser.add_argument('--seq-length', type=int, default=50,
                        help='input sequence length (default: 50)')
    parser.add_argument('--embedding-dim', type=int, default=256,
                        help='embedding dimension for characters in training model (default: 256)')
    parser.add_argument('--hidden-dim', type=int, default=256,
                        help='hidden state dimension in training model (default: 256)')
    parser.add_argument('--n-sent', type=int, default=50,
                        help='number of sentences to generate (default: 10)')
    return parser.parse_args()

def is_end(c):
    end_tokens=['。', '？', '！', '.', '?', '!']
    return c in end_tokens

def gen_text(model, dataset, args):
    model.eval()
    n = len(dataset)

    # Randomly choose a pattern to start text generation
    start = np.random.randint(0, n - 1)
    pattern=list(dataset[start][0].numpy())
    print(dataset[start][0])
    print(start)
    print(pattern)
    

    # Start generation until n_sent sentences generated 
    cnt = 0
    while cnt < args.n_sent: 
        # Format input pattern
        seq_in = torch.tensor(pattern, dtype=torch.long).reshape(1, -1)

        # Predict next character
        with torch.no_grad():
            pred = model(seq_in)
            pred = pred[-1] # last subsequence is the whole sequence
            prob = F.softmax(pred, dim=1)[0]
        char = np.random.choice(dataset.chars, p=prob.numpy()) # pick char based on probability instead of always picking the highest value
        char_idx = dataset.char_to_int[char]
        print(char, end='')

        # Append predicted character to pattern, remove first
        pattern.append(char_idx)
        pattern = pattern[1:]

        if is_end(char):
            cnt += 1 

def main():
    args = parse_args()

    # Load data
    dataset = TextDataset(args.corpus, seq_length=args.seq_length)

    # Load model
    model = Net(len(dataset.chars), args.embedding_dim, args.hidden_dim)
    model.load_state_dict(torch.load(args.output_model))

    # Generate text
    gen_text(model, dataset, args)

if __name__ == '__main__':
    main()
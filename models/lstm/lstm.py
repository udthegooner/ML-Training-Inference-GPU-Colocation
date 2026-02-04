# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
from tqdm import tqdm
import data
import model as mdl
from core.performanceIterator import PerformanceIterator

# --- Setup & Helpers ---

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def batchify(data, bsz, device):
    nbatch = data.size(0) // bsz
    data = data.narrow(0, 0, nbatch * bsz)
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)

def get_next_batch(source, i, bptt):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i+seq_len]
    target = source[i+1:i+1+seq_len].view(-1)
    return data, target

# --- Core Workloads ---

def train_model(args, model, corpus, device):
    train_data = batchify(corpus.train, args.batch_size, device)
    model.train()
    criterion = nn.NLLLoss()
    lr = args.lr
    
    # Ensure dataset is long enough for the requested num_steps
    while train_data.size(0) < args.num_steps * args.bptt:
        train_data = torch.cat([train_data, train_data], dim=0)

    # Standardize data_loader for PerformanceIterator
    data_range = range(0, train_data.size(0) - 1, args.bptt)
    if args.enable_perf_log:
        data_loader = PerformanceIterator(data_range, None, None, None, args.log_file)
    else:
        data_loader = data_range

    hidden = model.init_hidden(args.batch_size)
    total_loss = 0.
    start_time = time.time()
    
    for i, batch_start in enumerate(data_loader):
        if i >= args.num_steps:
            break
            
        data, targets = get_next_batch(train_data, batch_start, args.bptt)
        model.zero_grad()
        
        # LSTM specific: repackage hidden state to prevent backpropping through all time
        hidden = repackage_hidden(hidden)
        output, hidden = model(data, hidden)
        
        loss = criterion(output, targets)
        loss.backward()

        # Gradient clipping is vital for LSTMs to prevent exploding gradients
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        for p in model.parameters():
            p.data.add_(p.grad, alpha=-lr)

        total_loss += loss.item()
        
        if i % args.log_interval == 0 and i > 0:
            cur_loss = total_loss / args.log_interval
            print(f'| step {i:5d} | loss {cur_loss:5.2f} | ppl {math.exp(cur_loss):8.2f}')
            total_loss = 0
    total_time = time.time() - start_time
    print(f'Training time for {args.num_steps} steps: {total_time:.2f} seconds')
    print(f'Average training time per step: {total_time/args.num_steps:.4f} seconds')

def generate(args, device):
    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=device)
    model.eval()

    corpus = data.Corpus(args.data)
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(1)
    input = torch.randint(ntokens, (1, 1), dtype=torch.long).to(device)

    print(f"=> Generating {args.num_steps} words...")
    
    # Simple loop for inference benchmarking
    with torch.no_grad():
        for i in range(args.num_steps):
            output, hidden = model(input, hidden)
            word_weights = output.squeeze().div(args.temperature).exp().cpu()
            word_idx = torch.multinomial(word_weights, 1)[0]
            input.fill_(word_idx)
            
            if i % args.log_interval == 0:
                print(f'| Generated {i}/{args.num_steps} words')

# --- Entry Point ---

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='LSTM Wikitext-2 Benchmark')
    parser.add_argument('--gpuIdx', type=int, default=0)
    parser.add_argument('--job_type', type=str, choices=['training', 'inference'], default='training')
    parser.add_argument('--data', type=str, default='./data/wikitext-2')
    parser.add_argument('--model', type=str, default='LSTM')
    parser.add_argument('--lr', type=float, default=20)
    parser.add_argument('--clip', type=float, default=0.25)
    parser.add_argument('--batch_size', type=int, default=20)
    parser.add_argument('--bptt', type=int, default=35)
    parser.add_argument('--num_steps', type=int, default=1000)
    parser.add_argument('--log_interval', type=int, default=100)
    parser.add_argument('--enable_perf_log', action='store_true')
    parser.add_argument('--log_file', type=str, default="lstm.log")
    parser.add_argument('--checkpoint', type=str, default='./model.pt')
    parser.add_argument('--temperature', type=float, default=1.0)
    
    args = parser.parse_args()

    DEVICE = torch.device(f'cuda:{args.gpuIdx}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    if args.job_type == 'training':
        corpus = data.Corpus(args.data)
        model = mdl.RNNModel(args.model, len(corpus.dictionary), 200, 200, 2).to(DEVICE)
        train_model(args, model, corpus, DEVICE)
    else:
        generate(args, DEVICE)
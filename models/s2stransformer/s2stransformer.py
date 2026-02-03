# Ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse, random, numpy as np, torch, time, math
from typing import List
import torch.nn as nn
from torch import Tensor
from torch.nn import Transformer
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torchtext.datasets import multi30k, Multi30k

from core.performanceIterator import PerformanceIterator

# --- Setup & Config ---
SRC_LANGUAGE = 'de'
TGT_LANGUAGE = 'en'
UNK_IDX, PAD_IDX, BOS_IDX, EOS_IDX = 0, 1, 2, 3
DEVICE = 'cpu'

# Patching broken Multi30k URLs
multi30k.URL["train"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/training.tar.gz"
multi30k.URL["valid"] = "https://raw.githubusercontent.com/neychev/small_DL_repo/master/datasets/Multi30k/validation.tar.gz"

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic=True

# --- Model Architecture ---
# Adds positional encoding to the token embedding to introduce a notion of word order.
class PositionalEncoding(nn.Module):
    def __init__(self, embSize: int, dropout: float = 0.1, maxLen: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(0, maxLen).reshape(maxLen, 1)
        divTerm = torch.exp(torch.arange(0, embSize, 2) * (-math.log(10000) / embSize))
        posEmbedding = torch.zeros((maxLen, embSize))
        posEmbedding[:, 0::2] = torch.sin(position * divTerm)
        posEmbedding[:, 1::2] = torch.cos(position * divTerm)
        posEmbedding = posEmbedding.unsqueeze(-2)
        self.register_buffer('posEmbedding', posEmbedding)

    def forward(self, x: Tensor) -> Tensor:
        return self.dropout(x + self.posEmbedding[:x.size(0), :])

# Convert tensor of input indices into corresponding tensor of token embeddings
class TokenEmbedding(nn.Module):
    def __init__(self, vocabSize: int, embSize: int):
        super(TokenEmbedding, self).__init__()
        self.embedding = nn.Embedding(vocabSize, embSize)
        self.embSize = embSize

    def forward(self, tokens: Tensor) -> Tensor:
        return self.embedding(tokens.long()) * math.sqrt(self.embSize)

class Seq2SeqTransformer(nn.Module):
    def __init__(self, numEncoderLayers: int, numDecoderLayers: int,
                 embSize: int, nhead: int, srcVocabSize: int, tgtVocabSize: int,
                 dimFeedforward: int = 512, dropout: float = 0.1):
        super(Seq2SeqTransformer, self).__init__()
        self.transformer = Transformer(d_model=embSize,
                                       nhead=nhead,
                                       num_encoder_layers=numEncoderLayers,
                                       num_decoder_layers=numDecoderLayers,
                                       dim_feedforward=dimFeedforward,
                                       dropout=dropout)
        self.generator = nn.Linear(embSize, tgtVocabSize)
        self.srcTokEmbedding = TokenEmbedding(srcVocabSize, embSize)
        self.tgtTokEmbedding = TokenEmbedding(tgtVocabSize, embSize)
        self.positionalEncoding = PositionalEncoding(embSize, dropout=dropout)

    def forward(self, src: Tensor, tgt: Tensor,
                srcMask: Tensor, tgtMask: Tensor,
                srcPaddingMask: Tensor, tgtPaddingMask: Tensor,
                memoryKeyPaddingMask: Tensor) -> Tensor:
        srcEmb = self.positionalEncoding(self.srcTokEmbedding(src))
        tgtEmb = self.positionalEncoding(self.tgtTokEmbedding(tgt))
        outs = self.transformer(srcEmb, tgtEmb, srcMask, tgtMask, None,
                                srcPaddingMask, tgtPaddingMask, memoryKeyPaddingMask)
        return self.generator(outs)
    
    def encode(self, src: Tensor, srcMask: Tensor) -> Tensor:
        return self.transformer.encoder(self.positionalEncoding(
                            self.srcTokEmbedding(src)), srcMask)
    
    def decode(self, tgt: Tensor, memory: Tensor, tgtMask: Tensor) -> Tensor:
        return self.transformer.decoder(self.positionalEncoding(
                          self.tgtTokEmbedding(tgt)), memory,
                          tgtMask)

# --- Masking & Pipeline Helpers ---

def generateSquareSubsequentMask(sz: int) -> Tensor:
    mask = (torch.triu(torch.ones((sz, sz))) == 1).transpose(0, 1)
    mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
    return mask

def createMasks(src: Tensor, tgt: Tensor):
    srcSeqLen = src.shape[0]
    tgtSeqLen = tgt.shape[0]

    tgtMask = generateSquareSubsequentMask(tgtSeqLen)
    srcMask = torch.zeros((srcSeqLen, srcSeqLen), device=DEVICE).type(torch.bool)
    srcPaddingMask = (src == PAD_IDX).transpose(0, 1)
    tgtPaddingMask = (tgt == PAD_IDX).transpose(0, 1)
    return srcMask, tgtMask, srcPaddingMask, tgtPaddingMask

# club together sequential operations
def sequentialTransforms(*transforms):
    def func(txtInput):
        for transform in transforms:
            txtInput = transform(txtInput)
        return txtInput
    return func

# add BOS/EOS and create tensor for input sequence indices
def tensorTransform(token_ids: List[int]):
    return torch.cat((torch.tensor([BOS_IDX]),
                      torch.tensor(token_ids),
                      torch.tensor([EOS_IDX])))    

# --- Data Processing ---

token_transform = {
    'de': get_tokenizer('spacy', language='de_core_news_sm'),
    'en': get_tokenizer('spacy', language='en_core_web_sm')
}

def yieldTokens(dataIter, language):
    languageIndex = {SRC_LANGUAGE: 0, TGT_LANGUAGE: 1}
    for dataSample in dataIter:
        yield token_transform[language](dataSample[languageIndex[language]])

def getVocab():
    vocab = {}
    special_symbols = ['<unk>', '<pad>', '<bos>', '<eos>']
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        trainIter = Multi30k(split='train', language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
        vocab[ln] = build_vocab_from_iterator(yieldTokens(trainIter, ln),
                                             min_freq=1,
                                             specials=special_symbols,
                                             special_first=True)
        vocab[ln].set_default_index(UNK_IDX)
    return vocab

# collate data samples into batch tensors
def collateFn(batch, textTransform):
    srcBatch, tgtBatch = [], []
    for srcSample, tgtSample in batch:
        srcBatch.append(textTransform[SRC_LANGUAGE](srcSample.rstrip('\n')))
        tgtBatch.append(textTransform[TGT_LANGUAGE](tgtSample.rstrip('\n')))
    srcBatch = pad_sequence(srcBatch, padding_value=PAD_IDX)
    tgtBatch = pad_sequence(tgtBatch, padding_value=PAD_IDX)
    return srcBatch, tgtBatch

# --- Benchmarking ---
def runBenchmark(mode, model, dataloader, numSteps, optimizer=None, lossFn=None):
    model.to(DEVICE)
    model.train() if mode == 'training' else model.eval()
    totalLoss = 0.0
    correctPredictions = 0
    totalPredictions = 0
    totalTime = time.time()
    for i,(src,tgt) in enumerate(dataloader):
        if i>=numSteps:
            break
        src, tgt = src.to(DEVICE), tgt.to(DEVICE)
        tgtInput = tgt[:-1, :]

        srcMask, tgtMask, srcPaddingMask, tgtPaddingMask = createMasks(src, tgtInput)

        logits = model(src, tgtInput, srcMask, tgtMask, srcPaddingMask, tgtPaddingMask, srcPaddingMask)

        if mode == 'training':
            optimizer.zero_grad()
            tgtOut = tgt[1:, :]
            loss = lossFn(logits.reshape(-1, logits.shape[-1]), tgtOut.reshape(-1).long())
            totalLoss += loss.item()
            loss.backward()
            optimizer.step()
        elif mode == 'inference':
            predictedTokens = logits.argmax(dim=-1)
            correctPredictions += (predictedTokens == tgt[1:, :]).sum().item()
            totalPredictions += (tgt[1:, :] != PAD_IDX).sum().item()
        
        if mode == 'training' and i % 50 == 0:
            print(f'Step: {i}, {mode} loss per batch: {totalLoss / (i + 1):.4f}')

    totalTime = time.time() - totalTime
    print(f'Total {mode} time for {numSteps} steps: {totalTime:.2f} seconds')
    print(f'Average {mode} time per step: {totalTime/numSteps:.2f} seconds')
    if mode == 'inference':
        accuracy = correctPredictions / totalPredictions * 100
        print(f'Inference Accuracy: {accuracy:.2f}%')

# generate output sequence using greedy algorithm
def greedy_decode(model, src, src_mask, max_len, start_symbol):
    src = src.to(DEVICE)
    src_mask = src_mask.to(DEVICE)

    memory = model.encode(src, src_mask)
    ys = torch.ones(1, 1).fill_(start_symbol).type(torch.long).to(DEVICE)
    for i in range(max_len-1):
        memory = memory.to(DEVICE)
        tgt_mask = (generateSquareSubsequentMask(ys.size(0))
                    .type(torch.bool)).to(DEVICE)
        out = model.decode(ys, memory, tgt_mask)
        out = out.transpose(0, 1)
        prob = model.generator(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        next_word = next_word.item()

        ys = torch.cat([ys,
                        torch.ones(1, 1).type_as(src.data).fill_(next_word)], dim=0)
        if next_word == EOS_IDX:
            break
    return ys

# translate input sentence into target language
def translate(model: torch.nn.Module, src_sentence: str):
    start = time.time()
    src = text_transform[SRC_LANGUAGE](src_sentence).view(-1, 1)
    num_tokens = src.shape[0]
    src_mask = (torch.zeros(num_tokens, num_tokens)).type(torch.bool)
    tgt_tokens = greedy_decode(
        model,  src, src_mask, max_len=num_tokens + 5, start_symbol=BOS_IDX).flatten()
    totalTime = time.time() - start
    op = " ".join(vocab_transform[TGT_LANGUAGE].lookup_tokens(list(tgt_tokens.cpu().numpy()))).replace("<bos>", "").replace("<eos>", "")
    return op, totalTime


"""
pip install following
- torchtext
- spacy
- torchdata
- portalocker

run the following
 python3 -m spacy download de_core_news_sm
 python3 -m spacy download en_core_web_sm
"""
if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Transformer S2S Benchmark Workload')
    parser.add_argument('--gpuIdx', type=int, default=0, help='Index of GPU to use')
    parser.add_argument('--alpha', type=float, default=0.0005, help='Learning rate')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_steps', type=int, default=64) # number of batches to process
    parser.add_argument('--job_type', choices=['training', 'inference', 'translate'], required=True)
    parser.add_argument('--log_file', type=str)
    parser.add_argument('--enable_perf_log', action='store_false')
    parser.add_argument('--model_path', type=str, default=None, help='Path to save model')
    args = parser.parse_args()

    DEVICE = torch.device(f'cuda:{args.gpuIdx}' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {DEVICE}")

    set_seed(42)

    vocab_transform = getVocab()
    text_transform = {}
    for ln in [SRC_LANGUAGE, TGT_LANGUAGE]:
        text_transform[ln] = sequentialTransforms(token_transform[ln],
                                                  vocab_transform[ln],
                                                  tensorTransform)
    model = Seq2SeqTransformer(numEncoderLayers=4,
                                numDecoderLayers=4,
                                embSize=512,
                                nhead=8,
                                srcVocabSize=len(vocab_transform[SRC_LANGUAGE]),
                                tgtVocabSize=len(vocab_transform[TGT_LANGUAGE]),
                                )
    dataset = Multi30k(split='train' if args.job_type == 'training' else 'valid',
                       language_pair=(SRC_LANGUAGE, TGT_LANGUAGE))
    dataloader = DataLoader(dataset=dataset,
                            batch_size=args.batch_size,
                            collate_fn=lambda batch: collateFn(batch, text_transform))
    if args.enable_perf_log:
        dataloader = PerformanceIterator(dataloader, None, None, None, args.log_file)
    
    if args.job_type == 'training':
        optimizer = torch.optim.Adam(model.parameters(), lr=args.alpha, betas=(0.9, 0.98), eps=1e-9)
        lossFn = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
        runBenchmark('training', model, dataloader, args.num_steps, optimizer, lossFn)

        if args.model_path:
            torch.save(model.state_dict(), args.model_path)
            print(f"Model saved to {args.model_path}")
        
        model.eval()
        print(translate(model, "Sie spielen Fußball."))

    elif args.job_type == 'translate':
        if args.model_path is None:
            print("Model path not provided")
            exit()
        
        model.load_state_dict(torch.load(args.model_path))
        model = model.to(DEVICE)
        model.eval()

        while True:
            src_sentence = input("Enter a sentence in German: ")
            if src_sentence == 'exit':
                break
            translated_sentence, inferTime = translate(model, src_sentence)
            print(f"Translated sentence: {translated_sentence}")
            print(f"Inference time: {inferTime:.4f} seconds")
    else:
        if args.model_path is None:
            print("Model path not provided")
            exit()
        
        model.load_state_dict(torch.load(args.model_path))
        runBenchmark('inference', model, dataloader, args.num_steps)
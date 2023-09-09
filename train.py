from model import build_transformer
from dataset import causal_mask, BilingualDataset
from config import get_config, get_weights_file_path

import torchtext.datasets as datasets
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
from torch.optim.lr_scheduler import LambdaLR

import warnings
from tqdm import tqdm
import os
from pathlib import Path


# Huggingface datasets and tokenizers
from datasets import load_dataset, Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace

import torchmetrics
from torch.utils.tensorboard import SummaryWriter


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt, max_len, device):
    sos_idx = tokenizer_tgt.token_to_id('[SOS]')
    eos_idx = tokenizer_tgt.token_to_id('[EOS]')

    # Precompute the encoder output and reuse it for every step
    encoder_output = model.encode(source, source_mask)
    # Initialize the decoder input with the sos token
    decoder_input = torch.empty(1, 1).fill_(sos_idx).type_as(source).to(device)
    while True:
        if decoder_input.size(1) == max_len:
            break

        # Build mask for target
        decoder_mask = causal_mask(decoder_input.size(1)).type_as(source_mask).to(device)

        # Calculate the output
        out = model.decode(encoder_output, source_mask, decoder_input, decoder_mask)

        # Get next token
        prob = model.project(out[:, -1])
        _, next_word = torch.max(prob, dim=1)
        decoder_input = torch.cat(
            [decoder_input, torch.empty(1, 1).type_as(source).fill_(next_word.item()).to(device)], dim=1
        )

        if next_word == eos_idx:
            break

    return(decoder_input.squeeze(0))



def get_all_sentences(ds, lang):
    for item in ds:
        yield(item['translation'][lang])

def get_or_build_tokenizer(config, ds, lang):
    tokenizer_path = Path(config['tokenizer_file'].format(lang))
    if not Path.exists(tokenizer_path):
        # Most code taken from: https://huggingface.co/docs/tokenizers/quicktour
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"], min_frequency=2)
        tokenizer.train_from_iterator(get_all_sentences(ds, lang), trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))
    return(tokenizer)

'''def collate_fn(batch):
    encoder_inputs = [item["encoder_input"] for item in batch]
    decoder_inputs = [item["decoder_input"] for item in batch]

    tokens = batch[0]['tokens']
    SOS_TOKEN, EOS_TOKEN, PAD_TOKEN = tokens

    # Add [SOS] and [EOS] tokens to each sequence
    encoder_inputs = [torch.cat([torch.tensor([SOS_TOKEN], dtype=torch.int64), seq, torch.tensor([EOS_TOKEN], dtype=torch.int64)]) for seq in encoder_inputs]
    decoder_inputs = [torch.cat([torch.tensor([SOS_TOKEN], dtype=torch.int64), seq, torch.tensor([EOS_TOKEN], dtype=torch.int64)]) for seq in decoder_inputs]

    # Pad sequences within the batch
    encoder_inputs_padded = nn.utils.rnn.pad_sequence(encoder_inputs, batch_first=True, padding_value=PAD_TOKEN)
    decoder_inputs_padded = nn.utils.rnn.pad_sequence(decoder_inputs, batch_first=True, padding_value=PAD_TOKEN)

    # Create labels for the decoder. The label is the decoder input shifted by one position with [EOS] at the end.
    labels = [torch.cat([seq[1:], torch.tensor([EOS_TOKEN], dtype=torch.int64)]) for seq in decoder_inputs]
    labels_padded = nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=PAD_TOKEN)

    print('padding')
    print('encoder')
    print(encoder_inputs_padded.shape)
    print('encoder mask')
    print(((decoder_inputs_padded != PAD_TOKEN).unsqueeze(1).int() & causal_mask(decoder_inputs_padded.size(1))).shape)

    return {
        "encoder_input": encoder_inputs_padded,
        "decoder_input": decoder_inputs_padded,
        "encoder_mask": (encoder_inputs_padded != PAD_TOKEN).unsqueeze(1).int(),
        "decoder_mask": (decoder_inputs_padded != PAD_TOKEN).unsqueeze(1).int() & causal_mask(decoder_inputs_padded.size(1)),
        "label": labels_padded,
        "src_text": [item["src_text"] for item in batch],
        "tgt_text": [item["tgt_text"] for item in batch]
    }'''


def collate_fn(batch):
    train = True if len(batch) != 1 else False
    encoder_inputs, decoder_inputs = [], []
    encoder_masks, decoder_masks, labels = [], [], []
    src_texts, target_texts  = [], []

    max_en_batch_len = max(b['encoder_token_len'] for b in batch) + 2
    max_de_batch_len = max(b['decoder_token_len']  for b in batch) + 1

    # process
    for b in batch:
        # remove outliers or edge cases
        if train and (len(b['encoder_input']) < 2 or len(b['encoder_input']) > 150 or len(b['decoder_input']) >= len(b['encoder_input']) + 10):
            continue
            
        # dynamic padding
        enc_num_padding_tokens = max_en_batch_len - len(b['encoder_input']) # we will add <s> and </s>
        dec_num_padding_tokens = max_de_batch_len - len(b['decoder_input'])

        # Make sure the number of padding tokens is not negative. If it is, the sentence is too long
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("Sentence is too long")

        # Add <s> and </s> token
        encoder_input = torch.cat(
            [
                b['encoder_input'],
                torch.tensor([b['pad_token']] * enc_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        encoder_mask = (encoder_input != b['pad_token']).unsqueeze(0).unsqueeze(0).unsqueeze(0).int() # 1,1,seq_len

        # Add only </s> token
        label = torch.cat(
            [
                b['label'],
                torch.tensor([b['pad_token']] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )

        # Add only <s> token
        decoder_input = torch.cat(
            [
                b['decoder_input'],
                torch.tensor([b['pad_token']] * dec_num_padding_tokens, dtype=torch.int64)
            ],
            dim=0
        )
        decoder_mask = ((decoder_input != b['pad_token']).unsqueeze(0).int() & causal_mask(decoder_input.size(0))).unsqueeze(0)

        # append all data
        encoder_inputs.append(encoder_input)
        decoder_inputs.append(decoder_input)
        decoder_masks.append(decoder_mask)
        encoder_masks.append(encoder_mask)
        labels.append(label)
        src_texts.append(b['src_text'])
        target_texts.append(b['tgt_text'])

    return{
                "encoder_input": torch.vstack(encoder_inputs), 
                "decoder_input": torch.vstack(decoder_inputs), 
                "encoder_mask": torch.vstack(encoder_masks),
                "decoder_mask": torch.vstack(decoder_masks),
                "label": torch.vstack(labels), 
                "src_text": src_texts,
                "tgt_text": target_texts
            }


def get_ds(config):
    # It only has the train split, we divide ourselves
    ds_raw = load_dataset('opus_books', f"{config['lang_src']}-{config['lang_tgt']}", split='train')

    # Build tokenizers
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config['lang_src'])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config['lang_tgt'])

    # Filer out bad data
    ds_filtered = []
    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids

        if len(src_ids) < 2 or len(src_ids) > 150 or len(tgt_ids) > len(src_ids) + 10:
            continue
        
        ds_filtered.append(item)

    ds_raw = Dataset.from_list(ds_filtered)

    # Keep 90% for training, 10% for validation
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw, [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(train_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'])
    val_ds = BilingualDataset(val_ds_raw, tokenizer_src, tokenizer_tgt, config['lang_src'], config['lang_tgt'])

    # Find the maximum length of each sentence in the source and target sentence
    max_len_src = 0
    max_len_tgt = 0

    for item in ds_raw:
        src_ids = tokenizer_src.encode(item['translation'][config['lang_src']]).ids
        tgt_ids = tokenizer_tgt.encode(item['translation'][config['lang_tgt']]).ids
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

    print(f"Max length of source sentence: {max_len_src}")
    print(f"Max length of target sentence: {max_len_tgt}")

    train_dataloader = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True, collate_fn=collate_fn, num_workers=8)
    val_dataloader = DataLoader(val_ds, batch_size=1, shuffle=True, collate_fn=collate_fn)
    return(train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt)
    

def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(vocab_src_len, vocab_tgt_len, config['seq_len'], config['seq_len'], d_model=config['d_model'])
    return(model)

if __name__ == '__main__':
    warnings.filterwarnings("ignore")
    config = get_config()

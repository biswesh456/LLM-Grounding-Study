import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import math
import ast
import json
import textwrap
import re
import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random as rn
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import LlamaForCausalLM, LlamaTokenizerFast, Trainer, TrainingArguments,EarlyStoppingCallback, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorForLanguageModeling
import pathlib
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric
import os
from torch.optim import AdamW

print(torch.version.cuda, flush=True)
print(torch.cuda.is_available(), flush=True)
print("Total : ", torch.cuda.device_count(), flush=True)
print(torch.cuda.max_memory_allocated(), flush=True)

os.environ['WANDB_API_KEY'] = 'e819d741a0c770cb527b6ca091ee5d1b25a8222e'
os.environ['TRANSFORMERS_CACHE'] = '/home/bmohapat/.cache/huggingface/transformers'
os.environ['TORCH_EXTENSIONS_DIR'] = '/home/bmohapat/.cache/torch_extensions'


pattern = r'\[\d{2}:\d{2}:\d{2}\]'

# set seed
SEED = 3407
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'


def give_len_to_take(input_sen):
    return len(tokenizer(input_sen)['input_ids'])

def give_embeddings(sentences_list, instances_len):
    
    encoded_inputs = tokenizer.batch_encode_plus(sentences_list,
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
    
    # Retrieve the input tensors
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    
    batch_size = 1
    embeddings = []
    for i in tqdm(range(len(input_ids)), desc="Embeddings"):
        batch_input_ids = input_ids[i*batch_size:(i+1)*batch_size]
        batch_attention_mask = attention_mask[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            batch_embeddings = model(input_ids=batch_input_ids,attention_mask=batch_attention_mask, output_hidden_states=True).hidden_states[-1].detach().cpu().numpy()
#             batch_embeddings = model.(input_ids=batch_input_ids,attention_mask=batch_attention_mask).hidden_states[:,:,:].detach().cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    for index in range(len(embeddings)):
        embeddings[index] = np.mean(embeddings[index][-instances_len[index]:], axis=0)
        
    return embeddings


df = pd.read_csv("/home/bmohapat/github/LLM-Grounding-Study/data/Embedding_testing/expt_1_dataset_17_july_2_44_PM_GM_dialogs_removed.csv")

model_name = "/home/bmohapat/github/LLM-Grounding-Study/model/checkpoint-40" # here path to model comes
model = LlamaForCausalLM.from_pretrained(model_name).to(device)
model.config.use_cache = False
print(model.config)
print("Loading the tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained(
                "hf-internal-testing/llama-tokenizer",
                truncation_side="left",
                use_fast=False)

# Add pad token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})
    
df['d1_len'] = df['d1_instance_input_sentences'].apply(lambda x: give_len_to_take(x))
df['d2_len'] = df['d2_instance_input_sentences'].apply(lambda x: give_len_to_take(x))
df['d3_len'] = df['d3_instance_input_sentences'].apply(lambda x: give_len_to_take(x))
df['d4_len'] = df['d4_instance_input_sentences'].apply(lambda x: give_len_to_take(x))    

df['d1_instance_embeddings'] = give_embeddings(df['d1_input_sentences'].tolist(), df['d1_len'].tolist())
print("done with d1")
df['d2_instance_embeddings'] = give_embeddings(df['d2_paraphrased_dialogs'].tolist(), df['d2_len'].tolist())
print("done with d2")
df['d3_instance_embeddings'] = give_embeddings(df['d3_GPT3_paraphrased_dialogs'].tolist(), df['d3_len'].tolist())
print("done with d3")
df['d4_instance_embeddings'] = give_embeddings(df['d4_random_swap'].tolist(), df['d4_len'].tolist())
print("done with d4")
        
print("done with all the experiments")
df.to_json("expt_1_dataset_with_Llama_finetuned_embeddings.json")
    
    
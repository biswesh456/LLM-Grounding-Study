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
from transformers import T5ForConditionalGeneration, T5Tokenizer, Trainer, TrainingArguments,EarlyStoppingCallback, AutoModelForSeq2SeqLM, AutoTokenizer
from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric
import os
from torch.optim import AdamW


pattern = r'\[\d{2}:\d{2}:\d{2}\]'


SEED = 3407
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

df = pd.read_csv("/home/mkapadni/scratch/finetuning_experiments/EXPT_1_embeddings_comparison/expt_1_dataset_17_july_2_44_PM_GM_dialogs_removed.csv")

model_name = "/home/mkapadni/scratch/finetuning_experiments/T5_code/home/mkapadni/scratch/finetuning_experiments/models/t5_large_finetuned_meetup_d1/checkpoint-22880" # here path to model comes
#model_name = 't5-large'
model = T5ForConditionalGeneration.from_pretrained(model_name).to(device) # will uncomment in server
tokenizer = T5Tokenizer.from_pretrained(model_name, padding_side='right', truncation_side='left')

#model_name = 'microsoft/GODEL-v1_1-large-seq2seq'
#model_name = "/home/mkapadni/scratch/finetuning_experiments/T5_code/home/mkapadni/scratch/finetuning_experiments/models/godel_t5_large_finetuned_meetup_d1/checkpoint-22880"
#model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)
#tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right', truncation_side='left')


def give_len_to_take(input_sen):
    return len(tokenizer(input_sen)['input_ids'])

df['d1_len'] = df['d1_instance_input_sentences'].apply(lambda x: give_len_to_take(x))
df['d2_len'] = df['d2_instance_input_sentences'].apply(lambda x: give_len_to_take(x))
df['d3_len'] = df['d3_instance_input_sentences'].apply(lambda x: give_len_to_take(x))
df['d4_len'] = df['d4_instance_input_sentences'].apply(lambda x: give_len_to_take(x))


def give_embeddings(sentences_list, instances_len):
    
    encoded_inputs = tokenizer.batch_encode_plus(sentences_list,
            padding='max_length',
            truncation=True,
            max_length=2048,
            return_tensors='pt'
        )
    
    # Retrieve the input tensors
    input_ids = encoded_inputs['input_ids'].to(device)
    attention_mask = encoded_inputs['attention_mask'].to(device)
    
    batch_size = 8
    embeddings = []
    for i in tqdm(range(math.ceil(len(input_ids)/batch_size)), desc="Embeddings", total=math.ceil(len(input_ids)/batch_size)):
        batch_input_ids = input_ids[i*batch_size:(i+1)*batch_size]
        batch_attention_mask = attention_mask[i*batch_size:(i+1)*batch_size]
        with torch.no_grad():
            batch_embeddings = model.encoder(input_ids=batch_input_ids,attention_mask=batch_attention_mask).last_hidden_state[:,:,:].detach().cpu().numpy()
        embeddings.extend(batch_embeddings)
    
    for index in range(len(embeddings)):
        embeddings[index] = np.mean(embeddings[index][-instances_len[index]:], axis=0)
        
    return embeddings


df['d1_instance_embeddings'] = give_embeddings(df['d1_input_sentences'].tolist(), df['d1_len'].tolist())
print("done with d1")
df['d2_instance_embeddings'] = give_embeddings(df['d2_paraphrased_dialogs'].tolist(), df['d2_len'].tolist())
print("done with d2")
df['d3_instance_embeddings'] = give_embeddings(df['d3_GPT3_paraphrased_dialogs'].tolist(), df['d3_len'].tolist())
print("done with d3")
df['d4_instance_embeddings'] = give_embeddings(df['d4_random_swap'].tolist(), df['d4_len'].tolist())
print("done with d4")
        
print("done with all the experiments")
df.to_json("expt_1_dataset_with_T5_finetuned_embeddings.json")
    
    
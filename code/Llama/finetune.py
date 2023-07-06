import torch

from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random as rn
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import LlamaForCausalLM, LlamaTokenizerFast, Trainer, TrainingArguments,EarlyStoppingCallback, AutoModelForSeq2SeqLM, AutoTokenizer, DataCollatorWithPadding
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
# os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:4096'

SEED = 3407
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

train_df = pd.read_csv("../../data/meetup_d1_template_train_val/train_meetup_d1_template_short_image_descriptions.csv")
val_df = pd.read_csv("../../data/meetup_d1_template_train_val/val_meetup_d1_template_short_image_descriptions.csv")

print("train dataset shape", train_df.shape)
print("validation dataset shape", val_df.shape)
train_df.reset_index()
val_df.reset_index()

# Define custom dataset class
class TextDataset(Dataset):
    def __init__(self, tokenizer, texts, outputs):
        self.tokenizer = tokenizer
        self.texts = texts
        self.outputs = outputs
        self.cached_data_dict = {}

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        if index in self.cached_data_dict:
            return self.cached_data_dict[index]
        
        text = self.texts[index]
        output = self.outputs[index]
        tokenized_text = self.tokenizer(
            text + " " + output,
            padding='max_length',
            truncation=True,
            max_length=4096,
            return_tensors='pt'
        )

#         tokenized_output = self.tokenizer.encode_plus(
#             output,
#             padding='max_length',
#             truncation=True,
#             max_length=64,
#             return_tensors='pt'
#         )
        
        ret = {
            'input_ids': tokenized_text['input_ids'].squeeze(),
            'attention_mask': tokenized_text['attention_mask'].squeeze(),
            'decoder_input_ids': tokenized_text['input_ids'].clone().squeeze(),
            'labels': tokenized_text['input_ids'].clone().squeeze()
        }
        
        self.cached_data_dict[index] = ret
        
        return ret
    
# Define the training arguments
training_args = TrainingArguments(
    output_dir='/home/bmohapat/github/LLM-Grounding-Study/model',
    num_train_epochs=3,
    per_device_train_batch_size=2,
    per_device_eval_batch_size=1,
    gradient_accumulation_steps=32,
    warmup_steps=8,
    weight_decay=0.01,
    logging_dir='../../logs',
    logging_steps=1,
    evaluation_strategy='epoch',
    save_strategy='steps',
    save_steps=20,
    save_total_limit=10,
    learning_rate=2e-5,
    warmup_ratio=0.04,
    lr_scheduler_type="cosine",
    gradient_checkpointing=True,
    fp16=True,
    deepspeed='/home/bmohapat/github/LLM-Grounding-Study/code/Llama/deepspeed_config.json'
)


print("Loading the model...")
# tokenizer = LlamaTokenizer.from_pretrained("/home/bmohapat/pyllama_data/hf_weights/7B")
tokenizer = LlamaTokenizerFast.from_pretrained(
                "hf-internal-testing/llama-tokenizer",
                padding_side="right",
                truncation_side="left",
                use_fast=False)

model = LlamaForCausalLM.from_pretrained("/home/bmohapat/pyllama_data/hf_weights/7B")
model.config.use_cache = False
print(model.config)

# Add pad token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})

print("Creating the dataset...")
# Create train and validation datasets
train_dataset = TextDataset(tokenizer, train_df['inputs'].values.tolist(), train_df['outputs'].values.tolist())
val_dataset = TextDataset(tokenizer, val_df['inputs'].values.tolist(), val_df['outputs'].values.tolist())

print("created dataset")

# data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding='longest')

# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer
)

print("place_model_on_device", flush=True)
print(trainer.place_model_on_device, flush=True)
print("Max memory allocated cuda", torch.cuda.max_memory_allocated(), flush=True)

# Train the model
if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
    trainer.train(resume_from_checkpoint=True)
else:
    trainer.train()
trainer.save_state()


# Test the trained model
test_results = trainer.evaluate(eval_dataset=val_dataset)
print("Test Results:", test_results)

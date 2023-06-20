import torch

from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
import pandas as pd
import random as rn
import numpy as np
from sklearn.model_selection import train_test_split
from transformers import LlamaForCausalLM, LlamaTokenizer, Trainer, TrainingArguments,EarlyStoppingCallback, AutoModelForSeq2SeqLM, AutoTokenizer

from transformers.trainer_callback import EarlyStoppingCallback
from datasets import load_metric
import os
from torch.optim import AdamW

print(torch.cuda.is_available(), flush=True)
print(torch.cuda.device_count(), flush=True)

# os.environ['WANDB_API_KEY'] = 'e819d741a0c770cb527b6ca091ee5d1b25a8222e'

SEED = 3407
rn.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
device = 'cuda'

train_df = pd.read_csv("../../data/train_meetup_d1_template.csv")
val_df = pd.read_csv("../../data/val_meetup_d1_template.csv")

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

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        output = self.outputs[index]
        tokenized_text = self.tokenizer.encode_plus(
            text,
            padding='max_length',
            truncation=True,
            max_length=512,
            return_tensors='pt'
        )
        tokenized_output = self.tokenizer.encode_plus(
            output,
            padding='max_length',
            truncation=True,
            max_length=256,
            return_tensors='pt'
        )
        return {
            'input_ids': tokenized_text['input_ids'].squeeze(),
            'attention_mask': tokenized_text['attention_mask'].squeeze(),
            'decoder_input_ids': tokenized_output['input_ids'].squeeze(),
            'labels': tokenized_output['input_ids'].squeeze()
        }


print("Loading the model...")
tokenizer = LlamaTokenizer.from_pretrained("/home/bmohapat/pyllama_data/hf_weights/7B")
model = LlamaForCausalLM.from_pretrained("/home/bmohapat/pyllama_data/hf_weights/7B")

print("Creating the dataset...")
# Create train and validation datasets
train_dataset = TextDataset(tokenizer, train_df['inputs'].values.tolist(), train_df['outputs'].values.tolist())
val_dataset = TextDataset(tokenizer, val_df['inputs'].values.tolist(), val_df['outputs'].values.tolist())

# Define the training arguments
training_args = TrainingArguments(
    output_dir='./home/mkapadni/scratch/finetuning_experiments/models/godel_t5_large_finetuned_meetup_d1',
    num_train_epochs=5,
    per_device_train_batch_size=8,
    per_device_eval_batch_size=4,
    warmup_steps=500,
    weight_decay=0.01,
    logging_dir='./logs',
    logging_steps=100,
    evaluation_strategy='epoch',
    #eval_steps=500,
    save_strategy='epoch',
    save_total_limit=5,
    learning_rate=2e-5,
    load_best_model_at_end=True
)

# Define the ROUGE metric
rouge_metric = load_metric('rouge')

# Define the compute_metrics function for Trainer
def compute_metrics(eval_pred):
    predictions, labels = eval_pred.predictions, eval_pred.label_ids
    predictions = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    rouge = rouge_metric.compute(predictions=predictions, references=labels)
    rouge_scores = {key: value.mid.fmeasure * 100 for key, value in rouge.items()}
    return rouge_scores


# Create the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    #compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
)

# Train the model
trainer.train()

# Test the trained model
test_results = trainer.evaluate(eval_dataset=val_dataset)
print("Test Results:", test_results)



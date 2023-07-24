import torch

from torch import nn
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
from tqdm import tqdm
import math
import ast
import json
import re
import textwrap
import copy

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

def give_list(sent):
    res = ast.literal_eval(sent)
    return res

def give_input(instance, file_num, image_descriptions_dict, files_path, prior):
        get_all_utterances_before_this_index  = instance[0][0]
        get_final_instance_utterance = instance[-1][0]
        df = pd.read_csv(files_path+"dial_"+str(file_num)+".csv")
        
        convo_flow = []
        time_stamp = []
        a_image_description = []
        b_image_description = []
        index_current_iter = 0
        
        for convo,a_image, b_image, time, cur_index in zip(df['public'], df['A-private'], df['B-private'], df['time'], df['Unnamed: 0']):
                if (cur_index > int(get_final_instance_utterance)):
                        #print(cur_index, get_final_instance_utterance)
                        break
                
                if cur_index < int(get_all_utterances_before_this_index):
                        if (type(convo) == str) and (len(convo) > 2):
                                convo_flow.append(convo)
                                time_stamp.append(time[7:])
                        else:
                                convo_flow.append("")
                                time_stamp.append(time[7:])
                                
                        
                        if ((type(a_image)== str ) and ("url" in a_image)):
                                name = a_image.replace(" ", "")
                                name = name.split("/")[-1]
                                a_image_description.append(image_descriptions_dict[name])
                        else:
                                a_image_description.append("")

                        if ((type(b_image)== str ) and ("url" in b_image)):
                                name = b_image.replace(" ", "")
                                name = name.split("/")[-1]
                                b_image_description.append(image_descriptions_dict[name])
                        else:
                                b_image_description.append("")
                else:
                        break
                
                index_current_iter = index_current_iter + 1
                
        for i in range(len(instance)):
                convo_flow.append(instance[i][2])
                time_stamp.append(instance[i][1][7:])
                # get the index of the instance[i][0] in the "Unnamed: 0" column of the file
                index_curr = df[df['Unnamed: 0'] == int(instance[i][0])].index.values.astype(int)[0]
                a_image = df['A-private'][index_curr]
                b_image = df['B-private'][index_curr]
                if ((type(a_image)== str ) and ("url" in a_image)):
                        name = a_image.replace(" ", "")
                        name = name.split("/")[-1]
                        #a_image_description.append(image_descriptions_dict[name])
                        a_image_description.append("")
                else:
                        a_image_description.append("")

                if ((type(b_image)== str ) and ("url" in b_image)):
                        name = b_image.replace(" ", "")
                        name = name.split("/")[-1]
                        #b_image_description.append(image_descriptions_dict[name])
                        b_image_description.append("")
                else:
                        b_image_description.append("")
                        
        current_convo = prior +" "
        # WE NEED TO ADD ONLY THE LAST SEEN IMAGE OF A AND B TO THE DESCRIPTION
        # GET THE INDEX OF THE LAST SEEN IMAGE OF A AND B
        a_last_seen_index = 0
        b_last_seen_index = 0
        for i in range(len(convo_flow)):
                if a_image_description[i] != "":
                        a_last_seen_index = i
                        a_last_seen_image = a_image_description[i]
                if b_image_description[i] != "":
                        b_last_seen_index = i
                        b_last_seen_image = b_image_description[i]
                    
        flag_a = 0
        flag_b = 0
        for index in range(len(convo_flow)):
                
                if (index > a_last_seen_index) and (flag_a == 0):
                        flag_a = 1
                        current_convo = current_convo + " <Image A> " + a_last_seen_image + " " 
                        
                if (index > b_last_seen_index) and (flag_b == 0):
                        flag_b = 1
                        current_convo = current_convo + " <Image B> " + b_last_seen_image + " " 
                        
                
                if (convo_flow[index] != "A: ") and (convo_flow[index] != "B: ") and (convo_flow[index] != "GM: ") and (".w" not in convo_flow[index]) and (".e" not in convo_flow[index]) and (".n" not in convo_flow[index]) and (".s" not in convo_flow[index]) and (convo_flow[index] != "GM: ") and (len(convo_flow[index]) > 3):
                        current_convo = current_convo + " " + " ["+ time_stamp[index] + "] " + convo_flow[index] + " "
        
        current_convo = re.sub(' +', ' ', current_convo)
        return current_convo 

def calculate_perplexity(input_ids, labels, labels_wrong, tokenizer, trainer):
    inputs_c = tokenizer(input_ids + ' ' + labels, return_tensors='pt')
    inputs_w = tokenizer(input_ids + ' ' + labels_wrong, return_tensors="pt")
    
    response_c = tokenizer(labels, return_tensors='pt')
    response_w = tokenizer(labels_wrong, return_tensors='pt')
    
    labels_c = torch.zeros(inputs_c["input_ids"].shape).long() - 100
    labels_c[0, -response_c["input_ids"].shape[1]+1:] = response_c["input_ids"][0][1:]
    
    labels_w = torch.zeros(inputs_w["input_ids"].shape).long() - 100
    labels_w[0, -response_w["input_ids"].shape[1]+1:] = response_w["input_ids"][0][1:]
    
    with torch.no_grad():
        loss_c = trainer.predict({'input_ids' : inputs_c["input_ids"],'labels' : labels_c})
        print(loss_c)
        loss_w = trainer.predict({'input_ids' : inputs_w["input_ids"],'labels' : labels_w})
        ppl_c = torch.exp(loss_c).detach().to("cpu").item()
        ppl_w = torch.exp(loss_w).detach().to("cpu").item()
    return ppl_c, ppl_w

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
        outputs = self.outputs[index]
        
        inputs = tokenizer(text + ' ' + outputs, return_tensors='pt')
        response = tokenizer(outputs, return_tensors='pt')
        
        labels = torch.zeros(inputs["input_ids"].shape).long() - 100
        labels[0, -response["input_ids"].shape[1]+1:] = response["input_ids"][0][1:]

        ret = {
            'input_ids': inputs['input_ids'].squeeze(),
            'attention_mask': inputs['attention_mask'].squeeze(),
            'labels': labels.squeeze()
        }
        
        return ret


print("Loading the tokenizer...")
tokenizer = LlamaTokenizerFast.from_pretrained(
                "hf-internal-testing/llama-tokenizer",
                padding_side="right",
                truncation_side="left",
                use_fast=False)

print("Loading the model...")
model = LlamaForCausalLM.from_pretrained("/home/bmohapat/github/LLM-Grounding-Study/model/checkpoint-40")
model.config.use_cache = False
print(model.config)

# Add pad token if necessary
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({'pad_token': tokenizer.unk_token})

print("Creating the dataset...")

df = pd.read_csv("/home/bmohapat/github/LLM-Grounding-Study/data/perplexity/files_for_testing/cancel_d1_perplexity_testing_dataset_final.csv")

df['paraphrased']= df['paraphrased'].apply(lambda x: give_list(x))
df['choice_correct'] = df['choice_correct'].apply(lambda x: give_list(x))
df['choice_wrong'] = df['choice_wrong'].apply(lambda x: give_list(x))

print("shape of data", df.shape)

prior = "Instructions : Here is a conversation between two Participants A and B who are in different rooms. Each room has a type.  Participants are in one room at a time. They communicate with each other and describe their rooms. The participants can move to different rooms and describe their new room to the other participant until they reach the same room. The aim is to meet in the same room. The descriptions of the rooms that the participants currently saw are provided in text in between the utterances. The current room description of User A starts with a token <Image A> and the current room description of User B starts with a token <Image B>. Every utterance from A or B is preceded with a timestamp closed under brackets. GM is the third person who provides essential information regarding the game to both the participants."

#convert the images in image_path and description in description to a dict
files_path = "/home/bmohapat/github/LLM-Grounding-Study/data/original_dialogs/"
image_df = pd.read_csv("/home/bmohapat/github/LLM-Grounding-Study/data/image_descriptions_shortened_by_chatgpt.csv")

image_descriptions_dict = {}
for i in range(len(image_df)):
    image_descriptions_dict[image_df.iloc[i]['image_path']] = image_df.iloc[i]['summaries_shortened']

print("created test dataset")

print("create trainer")

# Define the arguments
testing_args = TrainingArguments(
    output_dir='/home/bmohapat/github/LLM-Grounding-Study/model',
    do_train = False,
    do_predict = True,
    per_device_eval_batch_size = 1,
    gradient_checkpointing=True,
    fp16=True,
    deepspeed='/home/bmohapat/github/LLM-Grounding-Study/code/Llama/deepspeed_config.json'
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# Create the Trainer
class CustomTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prediction_perplexity = []

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):
        # Run parent's method first
        outputs = super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        # The parent's method already computed the loss, we can just store it
        loss = outputs[0]
        
        pplx = torch.exp(loss).detach().to("cpu").item()
        self.prediction_perplexity.append(pplx)

        return outputs
    
trainer = CustomTrainer(
    model=model,
    args=testing_args,
    tokenizer=tokenizer,
    data_collator=data_collator
)

print("created trainer")

# Create dataset
print("creating dataset")
test_input_list = []
test_correct_output_list = []
test_wrong_output_list = []

for i in tqdm(range(len(df))):
    t = give_input(df['paraphrased'][i], df['File_name'][i], image_descriptions_dict, files_path, prior)
    correct = df['choice_correct'][i][2]
    wrong = df['choice_wrong'][i][2]
    
    test_input_list.append(t)
    test_correct_output_list.append(correct)
    test_wrong_output_list.append(wrong)
    
test_correct_dataset = TextDataset(tokenizer, test_input_list, test_correct_output_list)
test_wrong_dataset = TextDataset(tokenizer, test_input_list, test_wrong_output_list)

print("created dataset")

# Evaluate the model
correct_prediction = trainer.predict(test_correct_dataset)
print(correct_prediction[2])
correct_ppl = copy.deepcopy(trainer.prediction_perplexity)
print(correct_ppl)

wrong_prediction = trainer.predict(test_wrong_dataset)
print(wrong_prediction[2])
wrong_ppl = trainer.prediction_perplexity[len(correct_ppl):]
print(wrong_ppl)

print(len(correct_ppl), len(wrong_ppl))

df['ppl_correct'] = correct_ppl
df['ppl_wrong'] = wrong_ppl

# print statistics about the same two columsn
print("correct perplexity statistics")
print(df['ppl_correct'].describe())

print()
print("wrong perplexity statistics")
print(df['ppl_wrong'].describe())

df.to_csv("cancel_d1_perplexity_testing_dataset_final_checked_with_llama_finetuned.csv")

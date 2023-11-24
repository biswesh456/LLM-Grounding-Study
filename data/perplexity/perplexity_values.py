import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import math
import ast
import json
import re
import torch
import textwrap
pattern = r'\[\d{2}:\d{2}:\d{2}\]'
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

pattern = r'\[\d{2}:\d{2}:\d{2}\]'

def give_list(sent):
    res = ast.literal_eval(sent)
    return res

device = "cuda"

df = pd.read_csv("/home/mkapadni/scratch/finetuning_experiments/EXPT_1_embeddings_comparison/cancel_d1_perplexity_testing_dataset_final.csv")
df.head()


df['paraphrased']= df['paraphrased'].apply(lambda x: give_list(x))
df['choice_correct'] = df['choice_correct'].apply(lambda x: give_list(x))
df['choice_wrong'] = df['choice_wrong'].apply(lambda x: give_list(x))

prior = "Instructions : Here is a conversation between two Participants A and B who are in different rooms. Each room has a type.  Participants are in one room at a time. They communicate with each other and describe their rooms. The participants can move to different rooms and describe their new room to the other participant until they reach the same room. The aim is to meet in the same room. The descriptions of the rooms that the participants currently saw are provided in text in between the utterances. The current room description of User A starts with a token <Image A> and the current room description of User B starts with a token <Image B>. Every utterance from A or B is preceded with a timestamp closed under brackets. GM is the third person who provides essential information regarding the game to both the participants."
#model_name = "/home/mkapadni/scratch/finetuning_experiments/T5_code/home/mkapadni/scratch/finetuning_experiments/models/t5_large_finetuned_meetup_d1/checkpoint-22880" # here path to model comes
model_name = "/home/mkapadni/scratch/finetuning_experiments/T5_code/home/mkapadni/scratch/finetuning_experiments/models/godel_t5_large_finetuned_meetup_d1/checkpoint-22880"
#model_name = 'microsoft/GODEL-v1_1-large-seq2seq'
#model_name = 't5-large'
#model = T5ForConditionalGeneration.from_pretrained(model_name).to(device) # will uncomment in server
#tokenizer = T5Tokenizer.from_pretrained(model_name, padding_side='right', truncation_side='left')

tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='right', truncation_side='left')
model = AutoModelForSeq2SeqLM.from_pretrained(model_name).to(device)

files_path = "/home/bmohapat/github/LLM-Grounding-Study/data/original_dialogs/"
image_df = pd.read_csv("home/bmohapat/github/LLM-Grounding-Study/data/image_descriptions_shortened_by_chatgpt.csv")

#convert the images in image_path and description in description to a dict
image_descriptions_dict = {}
for i in range(len(image_df)):
    image_descriptions_dict[image_df.iloc[i]['image_path']] = image_df.iloc[i]['summaries_shortened']


def give_input(instance, file_num, image_descriptions_dict):
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
                
        #for utterance in instance:
        #        current_convo = current_convo + " " + " ["+ utterance[1][7:] + "] " + utterance[2] + " "
        
        current_convo = re.sub(' +', ' ', current_convo)
        return current_convo 

#t = give_input(df['paraphrased'][0], df['File_name'][0], image_descriptions_dict)

def calculate_perplexity(input_ids, labels, labels_wrong):
    inputs = tokenizer(input_ids, return_tensors="pt")
    
    labels_c = tokenizer(labels,return_tensors='pt')
    labels_w = tokenizer(labels_wrong, return_tensors="pt")
    
    loss_c = model(input_ids=inputs["input_ids"].to(device),labels=labels_c["input_ids"].to(device)).loss
    loss_w = model(input_ids=inputs["input_ids"].to(device),labels=labels_w["input_ids"].to(device)).loss
    
    ppl_c = torch.exp(loss_c).to("cpu").item()
    ppl_w = torch.exp(loss_w).to("cpu").item()
    return ppl_c, ppl_w


ppl_c = []
ppl_w = []

for i in tqdm(range(len(df))):
    t = give_input(df['paraphrased'][i], df['File_name'][i], image_descriptions_dict)
    correct = df['choice_correct'][i][2]
    wrong = df['choice_wrong'][i][2]
    c,w = calculate_perplexity(t, correct, wrong)
    ppl_c.append(c)
    ppl_w.append(w)
    

df['ppl_correct'] = ppl_c
df['ppl_wrong'] = ppl_w

# print statistics about the same two columsn
print("correct perplexity statistics")
print(df['ppl_correct'].describe())

print()
print("wrong perplexity statistics")
print(df['ppl_wrong'].describe())

df.to_csv("cancel_d1_perplexity_testing_dataset_final_checked_with_perplexity_godel_t5_finetuned.csv")





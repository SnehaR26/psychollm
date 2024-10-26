## Imports
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "1" 
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import transformers
import torch
import pandas as pd
import csv
import sys
import io
import re


def promptfunc1(person,text,likert_scale):

    messages = [
                {"role": "system", "content": f"You are {person}. Respond strictly with a single number."},
                {"role": "user", "content": f"Choose one option from: {', '.join(likert_scale)} to rate the following statement: I see myself as someone who {text}. Respond ONLY with a single number between 1 and 5. You must not include any other text, words, or explanations in your response."}
            ]
            
    outputs = pipeline(
                messages,
                max_new_tokens=20,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.85,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                batch_size=4,
            )
            
    generated_text = outputs[0]['generated_text'][-1]['content']
    return generated_text

def promptfunc2(person,text,likert_scale):

    messages = [
                {"role": "system", "content": f"You are {person}. Respond strictly with a single number."},
                {"role": "user", "content": f"Rate the following statement: I see myself as someone who {text}. Choose one option from: {', '.join(likert_scale)}. Respond ONLY with a single number between 1 and 5. You must not include any other text, words, or explanations in your response."}
            ]
            
    outputs = pipeline(
                messages,
                max_new_tokens=20,
                do_sample=True,
                top_k=50,
                top_p=0.9,
                temperature=0.85,
                num_return_sequences=1,
                eos_token_id=tokenizer.eos_token_id,
                batch_size=4,
            )
            
    generated_text = outputs[0]['generated_text'][-1]['content']
    return generated_text

def runpipeline(model, scale, subjectlist, promptfunc, ptest, resfile):
    """
    promptfunc: 1 - bfi prompt1 
    2- bfi prompt2 
    3- sd3 prompt1 
    4- sd3 prompt2
    """    
    ## Setup environment ####

    # Set the CUDA_VISIBLE_DEVICES environment variable
    #os.environ['CUDA_VISIBLE_DEVICES'] = cudadev
    # Verify if it's set correctly
    print(os.environ['CUDA_VISIBLE_DEVICES'])     
    print(torch.cuda.is_available())  
    #device = torch.device(f"cuda:1" if torch.cuda.is_available() else "cpu")
    #print(f"Using device: {device}")
    tokenizer = AutoTokenizer.from_pretrained(model)
    #model = AutoModelForCausalLM.from_pretrained(model).to(device)
    pipeline = transformers.pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    torch_dtype=torch.bfloat16,
    device_map="auto",

    )
    ### data 


    Ques44 = ["Is talkative",
        "Tends to find fault with others",
        "Does a thorough job",
        "Is depressed, blue",
        "Is original, comes up with new ideas",
        "Is reserved",
        "Is helpful and unselfish with others",
        "Can be somewhat careless",
        "Is relaxed, handles stress well",
        "Is curious about many different things",
        "Is full of energy",
        "Starts quarrels with others",
        "Is a reliable worker",
        "Can be tense",
        "Is ingenious, a deep thinker",
        "Generates a lot of enthusiasm",
        "Has a forgiving nature",
        "Tends to be disorganized",
        "Worries a lot",
        "Has an active imagination",
        "Tends to be quiet",
        "Is generally trusting",
        "Tends to be lazy",
        "Is emotionally stable, not easily upset",
        "Is inventive",
        "Has an assertive personality",
        "Can be cold and aloof",
        "Perseveres until the task is finished",
        "Can be moody",
        "Values artistic, aesthetic experiences",
        "Is sometimes shy, inhibited",
        "Is considerate and kind to almost everyone",
        "Does things efficiently",
        "Remains calm in tense situations",
        "Prefers work that is routine",
        "Is outgoing, sociable",
        "Is sometimes rude to others",
        "Makes plans and follows through with them",
        "Gets nervous easily",
        "Likes to reflect, play with ideas",
        "Has few artistic interests",
        "Likes to cooperate with others",
        "Is easily distracted",
        "Is sophisticated in art, music, or Literature"]
    valid_set = set(['1', '2', '3', '4', '5'])
    Ques10= [
        "is reserved",
        "is generally trusting",
        "tends to be lazy",
        "is relaxed, handles stress well",
        "has few artistic interests",
        "is outgoing, sociable",
        "tends to find fault with others",
        "does a thorough job",
        "gets nervous easily",
        "has an active imagination"
    ]
    Ques27 = [
    "It's not wise to tell your secrets",
    "I like to use clever manipulation to get my way",
    "Whatever it takes, you must get the important people on your side",
    "Avoid direct conflict with others because they may be useful in the future",
    "It's wise to keep track of information that you can use against people later",
    "You should wait for the right time to get back at people",
    "There are things you should hide from other people to preserve your reputation",
    "Make sure your plans benefit yourself, not others",
    "Most people can be manipulated",
    "People see me as a natural leader",
    "I hate being the center of attention",
    "Many group activities tend to be dull without me",
    "I know that I am special because everyone keeps telling me so",
    "I like to get acquainted with important people",
    "I feel embarrassed if someone compliments me",
    "I have been compared to famous people",
    "I am an average person",
    "I insist on getting the respect I deserve",
    "I like to get revenge on authorities",
    "I avoid dangerous situations",
    "Payback needs to be quick and nasty",
    "People often say I'm out of control",
    "It's true that I can be mean to others",
    "People who mess with me always regret it",
    "I have never gotten into trouble with the law",
    "I enjoy having sex with people I hardly know",
    "I'll say anything to get what I want"
    ]
    
    name_category_dict = dict(zip(subjectlist['Name'], subjectlist['source']))

    ###################################running pipeline#######################################################
    gcounter = 0
    #multiattempt=15
    max_retries = 20
    #for loopfive in range(multiattempt): 
        
    for name, category in name_category_dict.items():
        itemcount = 1
        for text in Ques27:
            
            retry_count = 0
            valid_result = False
            pcounter=0
            while not valid_result and retry_count < max_retries:
                print(name,text)
                gcounter=gcounter+1
                pcounter=pcounter+1
                if promptfunc==1 and ptest=="bfi":
                    messages = [
                                {"role": "system", "content": f"You are {name}. Respond strictly with a single number."},
                                {"role": "user", "content": f"Choose one option from: {', '.join(scale)} to rate the following statement: I see myself as someone who {text}. Respond ONLY with a single number between 1 and 5. You must not include any other text, words, or explanations in your response."}
                                ]
            
                    outputs = pipeline(
                    messages,
                    max_new_tokens=20,
                    do_sample=True,
                    top_k=50,
                    top_p=0.9,
                    temperature=0.85,
                    num_return_sequences=1,
                    eos_token_id=tokenizer.eos_token_id,
                    batch_size=4,
                    )
            
                    generated_text = outputs[0]['generated_text'][-1]['content']
                    
                else:
                    generated_text = promptfunc2(name,text,scale)
                match = re.fullmatch(r'^[^1-5]*([1-5])[^1-5]*$', generated_text)
                print(generated_text)
                if match:
                    valid_result = True
                    print(match.group(1))
                    with open(resfile, 'a', newline='') as csvfile:
                        print("counter was here")
                        writer = csv.writer(csvfile)
                        writer.writerow([name,category,text,itemcount,match.group(1)])
                    print("Appended valid result to CSV")# Print the valid number
                    itemcount=itemcount+1
                else:
                    retry_count += 1
                    print(f"Invalid response. Retrying... (Attempt {retry_count}/{max_retries})")

            if not valid_result:
                print(f"Error: Maximum retries reached for question: '{text}'")
                with open(resfile, 'a', newline='') as csvfile:
                        #print("counter was here")
                        writer = csv.writer(csvfile)
                        writer.writerow([name,category,text,itemcount])
                print("Appended empty row to CSV")# Print the valid number
                itemcount=itemcount+1
            print("=======================================================================================================")
            
            print(f"It took {pcounter} retries for {name} for this question {text}")
    print(f"It took max {gcounter} tries")




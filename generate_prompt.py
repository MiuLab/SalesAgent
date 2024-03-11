import argparse
from tqdm import tqdm
import time
from json import dumps, dump
import json
import os
import pprint
import openai
import random
from src.utils import *

os.environ["OPENAI_API_KEY"] = "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"
openai.api_key = os.getenv("OPENAI_API_KEY")


def Print_Intent():
    msgd = read_json_file(MSGD_PATH)
    stod = read_json_file(STOD_PATH)
    intent_set = set()
    for i, d in enumerate(msgd):
        for intent in d['intent']['type']:
            intent_set.add(intent)
    for i, d in enumerate(stod):
        for intent in d['intent']['type']:
            intent_set.add(intent)
    print(intent_set)
        
    
def generate_rewrite_prompt(datasets):
    PATH = MSGD_PATH if datasets == "MSGD" else STOD_PATH
    files = read_json_file(PATH)
    prompts = []
    # generate prompts for MSGD
    for i, d in enumerate(files):
        # concat all the dialogue with User: or Agent: in the front
        # if the intent position is a even number, then the first speaker is user, otherwise it's the agent
        dialogue = ''
        transition = d['transition_candidates']
        for j, s in enumerate(d['dialog']):
            if s in transition:
                break
            if j % 2 == d['intent']['position']%2:
                dialogue += 'User: ' + s + "\n"
            else:
                dialogue += 'Agent: ' + s + "\n"

        dict = {}
        prompt = REWRITE_PREFIX  + dialogue + '\n' +REWRITE_SUFFIX
        dict['id'] = d['id']
        dict['context'] = dialogue
        dict['prompt'] = prompt
        prompts.append(dict)
    write_json_file(f"./data/dialogues/{datasets}_prompts_for_rewrite.json", prompts)

    # Write the prompts to txt file
    prompts_txt = ''
    for i, d in enumerate(prompts):
        prompts_txt += d['id'] + "\n" + d['prompt'] + "\n"
    write_txt_file(f"./data/dialogues/{datasets}_prompts.txt", prompts_txt)


def generate_intent_det_prompt(datasets):
    REWRITE_PATH = MSGD_REWRITE_PATH if datasets == "MSGD" else STOD_REWRITE_PATH
    files = []
    file_ls = os.listdir(REWRITE_PATH)
    prompts = []
    fail_id_ls = []
    for f in file_ls:
        if f[-5:] == '.json':
            files.append(read_json_file(REWRITE_PATH+'/'+f))
    for i, d in enumerate(files):
        try:
            rewritten_dialog = parse_response(d['response'])
        except:
            fail_id_ls.append(d['id'])
            continue

        intent_detection_prompt = INTENT_DET_PREFIX + "\n\n" + rewritten_dialog + "\n\n"+ INTENT_DET_SUFFIX
        dic = {}
        dic['id'] = d['id']
        dic['context'] = rewritten_dialog
        dic['prompt'] = intent_detection_prompt
        prompts.append(dic)
    print(f"Fail rewrite_id:{fail_id_ls} with len {len(fail_id_ls)}")
    write_json_file(f"./data/dialogues/{datasets}_prompts_for_intent_detection_final.json", prompts)

    # Write the prompts to txt file
    prompts_txt = ''
    for i, d in enumerate(prompts):
        prompts_txt += d['id'] + "\n" + d['prompt'] + "\n"

    write_txt_file(f"./data/dialogues/{datasets}_prompts_for_intent_detection_final.txt", prompts_txt)


def generate_continue_prompt(datasets):
    INTENT_PATH = MSGD_INTENT_PATH if datasets == "MSGD" else STOD_INTENT_PATH
    prompts = read_json_file(f"./data/dialogues/{datasets}_prompts_for_intent_detection_final.json")
    id2response = {}
    file_ls = os.listdir(INTENT_PATH)
    continue_prompts = []
    non_exist_intent_id = []
    cnt_dic = {}
    fail_id_ls = []
    len_context_eq_1 = []
    for f in file_ls:
        if f[-5:] == '.json':
            response = read_json_file(INTENT_PATH+'/'+f)
            id2response[response['id']] = response
    for key in intent_dic.keys():
        cnt_dic[key] = 0
    for i, d in enumerate(prompts):
        rewritten_dialog = d['context']
        if len(rewritten_dialog.split('\n')) == 1:
            len_context_eq_1.append(d['id'])
            continue
        dic = {}
        try:
            intent, description = parse_intent(id2response[d['id']]['response'])
        except:
            fail_id_ls.append(d['id'])
            continue
        if description == '':
            dic['description'] = "Created by LLM"
            non_exist_intent_id.append(d['id'])
            continue
        else:
            dic['description'] = description
        continue_prompt = CONTINUE_PREFIX + intent + "\n\n" + rewritten_dialog + "\n\n" + CONTINUE_SUFFIX
        dic['id'] = d['id']
        dic['intent'] = intent
        dic['prompt'] = continue_prompt
        dic['context'] = d['context']
        if intent in list(cnt_dic.keys()):
            cnt_dic[intent] +=1
        continue_prompts.append(dic)
    write_json_file(f"./data/dialogues/{datasets}_prompts_for_continue_final.json", continue_prompts)
    write_json_file(f"./data/dialogues/one_turn_chitchat_id.json",len_context_eq_1)
    write_json_file(f"./data/dialogues/non_exist_intent_id.json",non_exist_intent_id)

    # Write the prompts to txt file
    prompts_txt = ''
    for i, d in enumerate(continue_prompts):
        prompts_txt += d['id'] + "\n" + d['prompt'] + "\n"
    write_txt_file(f"./data/dialogues/{datasets}_prompts_for_continue_final.txt", prompts_txt)
    print(f"Number of intent that does not exist:{len(non_exist_intent_id)}")
    print(cnt_dic)
    print(f"Fail rewrite_id:{fail_id_ls} with len {len(fail_id_ls)}")
    print(f"Number of dialog context with 1 turn:{len(len_context_eq_1)}")
    print(f"Total num of data:{len(continue_prompts)}")

def generate_trans_det_prompt(datasets):
    CONTINUE_PATH = MSGD_CONTINUE_PATH if datasets == "MSGD" else STOD_CONTINUE_PATH
    prompts = read_json_file(f"./data/dialogues/{datasets}_prompts_for_continue_final.json")
    prompts_one = read_json_file(f"./data/dialogues/{datasets}_prompts_for_continue_final_one_turn.json")
    file_ls = os.listdir(CONTINUE_PATH)
    trans_det_prompts = []
    id2response = {}
    fail_id_ls = []
    for f in file_ls:
        if f[-5:] == '.json':
            response = read_json_file(CONTINUE_PATH+'/'+f)
            id2response[response['id']] = response
    for i, d in enumerate(prompts):
        rewritten_dialog = d['context']
        intent = d['intent']
        try:
            continued_dialog = parse_continue(id2response[d['id']]['response'])
        except:
            fail_id_ls.append(d['id'])

        if_overlap = 0
        continued_dialog_ls = continued_dialog.split("\n")
        for i,s in enumerate(continued_dialog_ls):
            s  = s.strip()
            if s != ""  and s in rewritten_dialog:
                continued_dialog_ls.pop(i)
        trans_det_prompt = ""
        continued_dialog = "\n".join(continued_dialog_ls)
        dic = {}
        trans_det_prompt = TRANS_DET_PREFIX + "\n\n" + "Intent: "+ intent + "\n\n" + "Dialogue:\n" + rewritten_dialog + "\n" + continued_dialog +"\n\n" + TRANS_DET_SUFFIX
        dic['continued_dialogue'] = rewritten_dialog+"\n"+ continued_dialog

        dic['id'] = d['id']
        dic['intent'] = intent
        dic['prompt'] = trans_det_prompt
        dic['context'] = d['context']
        dic['description'] = d['description']
        trans_det_prompts.append(dic)
    for i, d in enumerate(prompts_one):
        rewritten_dialog = d['context']
        intent = d['intent']
        try:
            continued_dialog = parse_continue(id2response[d['id']]['response'])
        except:
            fail_id_ls.append(d['id'])
        if_overlap = 0
        continued_dialog_ls = continued_dialog.split("\n")
        for i,s in enumerate(continued_dialog_ls):
            s  = s.strip()
            if s != ""  and s in rewritten_dialog:
                continued_dialog_ls.pop(i)
        trans_det_prompt = ""
        continued_dialog = "\n".join(continued_dialog_ls)
        dic = {}
        trans_det_prompt = TRANS_DET_PREFIX + "\n\n" + "Intent: "+ intent + "\n\n" + "Dialogue:\n" + rewritten_dialog + "\n" + continued_dialog +"\n\n" + TRANS_DET_SUFFIX
        dic['continued_dialogue'] = rewritten_dialog+"\n"+ continued_dialog

        dic['id'] = d['id']
        dic['intent'] = intent
        dic['prompt'] = trans_det_prompt
        dic['context'] = d['context']
        dic['description'] = d['description']
        trans_det_prompts.append(dic)
    print(f"Num of data: {len(trans_det_prompts)}")
    print(fail_id_ls)
    print(f"Num of fail parse:{len(fail_id_ls)}")
    # write_json_file(f"./data/dialogues/{datasets}_prompts_for_trans_det.json", trans_det_prompts)
    prompts_txt = ''
    for i, d in enumerate(trans_det_prompts):
        prompts_txt += d['id'] + "\n" + d['prompt'] + "\n"
    # write_txt_file(f"./data/dialogues/{datasets}_prompts_for_trans_det.txt", prompts_txt)
    write_txt_file(f"test.txt", prompts_txt)


def generate_dataset(datasets):
    TRANS_PATH = MSGD_TRANS_PATH if datasets == "MSGD" else STOD_TRANS_PATH
    id2response = {}
    prompts = read_json_file(f"./data/dialogues/{datasets}_prompts_for_trans_det.json")
    file_ls = os.listdir(TRANS_PATH)
    data = []
    fail_id_ls = []
    fail_detect_id_ls = []
    total_len = []
    chitchat_len = []
    transition_len = []
    for f in file_ls:
        if f[-5:] == '.json':
            response = read_json_file(TRANS_PATH+'/'+f)
            id2response[response['id']] = response

    for i, d in enumerate(prompts):
        context = []
        for s in d['context'].split('\n'):
            try:
                import re
                speaker = re.search(r'(User|Agent): (.*)',s).group(1)
                s = re.search(r'(User|Agent): (.*)',s).group(2)
                s = s.strip("\"")
                s = s.strip("\'")
                s = speaker + ":" +" " + s
                context.append(s)
            except:
                continue
        prev_speaker = ''
        continued = []
        for s in d['continued_dialogue'].split('\n'):
            try:
                import re
                speaker = re.search(r'(User|Agent): (.*)',s).group(1)
                if prev_speaker == speaker:
                    continue
                s = re.search(r'(User|Agent): (.*)',s).group(2)
                s = s.strip("\"")
                s = s.strip("\'")
                s = speaker + ":" +" " + s
                continued.append(s)
                prev_speaker = speaker
            except:
                continue
        try:
            transition, position = parse_transition(id2response[d['id']]['response'],"\n".join(continued))
        except:
            fail_id_ls.append(d['id'])
            continue
        if position == -1:
            fail_detect_id_ls.append(d['id'])
            continue


        dic = {}
        dic['id'] = d['id']
        dic['intent'] = {
            "type": d['intent'],
            "description": d['description']
        }
        dic['transition_sentence'] = {
            "utterance": transition,
            "position": position
        }
        dic['chitchat_context'] = context
        chitchat_len.append(len(context))
        dic['dialog'] = continued
        total_len.append(position+1)
        trans = position+1 -len(context)
        trans = trans if trans > 0 else 0
        transition_len.append(trans)
        data.append(dic)


    # sort list of dictionary by id
    data = sorted(data, key=lambda i: i['id'])
    print(f"Num of data: {len(data)}")
    print(f"Num of fail parse:{len(fail_id_ls)}")
    print(f"Num of fail detect:{len(fail_detect_id_ls)}")
    print(f"avg turns of total:{sum(total_len)/len(total_len)}")
    print(f"avg turns of chit_chat:{sum(chitchat_len)/len(chitchat_len)}")
    print(f"avg turns of trans:{sum(transition_len)/len(transition_len)}")
    write_json_file(f"./data/dialogues/{datasets}_dataset_final.json", data)


def generate_from_LLM(datasets, task, input_file, output_dir):
    import threading
    threads = []

    prompts = read_json_file(input_file)
    os.makedirs(output_dir+"/"+datasets, exist_ok=True)

    # Detect intent for msgd rewritten dialogue

    # progress bar showing number of iteration and id
    progress_bar = tqdm(prompts)

    for i, d in enumerate(progress_bar):
        progress_bar.set_description(f"Task: {task} -- Iteration: {i} -- Dialogue ID: {d['id']}")
        if len(threads) % 30 == 0:
            time.sleep(60)
        thread = threading.Thread(target=get_response, args=(d['prompt'],task,datasets,d,output_dir))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()
    
    print(f"Num of data{len(threads)}")
    print(f"Finish Task: {task} for {datasets}")

def LLM_filter(datasets, input_file, output_dir):
    print(input_file)
    prompts = read_json_file(input_file)
    task = "Filter"
    os.makedirs(output_dir+"/"+datasets, exist_ok=True)
    prompts_new = []
    for i, d in enumerate(prompts):
        dialogue = ""
        for j, u in enumerate(d['dialog']):
            if  j == d['transition_sentence']['position']+1:
                break
            dialogue += u + "\n"
        # import random
        # intent = random.choice(list(intent_dic.keys()))
        intent = d['intent']['type']
        SUFFIX = f"1. Does the user show the intent of {intent}?\n \
2.Is it reasonable if the agent suggest anything partially related to the intent {intent}?\n \
You should only reply yes, no and why."
        d['prompt'] = dialogue + '\n' + SUFFIX
        prompts_new.append(d)

    import threading
    threads = []
    progress_bar = tqdm(prompts)

    for i, d in enumerate(progress_bar):
        progress_bar.set_description(f"Task: {task} -- Iteration: {i} -- Dialogue ID: {d['id']}")
        if len(threads) % 100 == 0:
            time.sleep(30)
        thread = threading.Thread(target=get_response, args=(d['prompt'],task,datasets,d,output_dir))
        thread.start()
        threads.append(thread)

    for thread in threads:
        thread.join()

    print(f"Num of data{len(threads)}")
    print(f"Finish Task: {task} for {datasets}")

def filter_out(datasets):
    FILTER_PATH = MSGD_FILTER_PATH if datasets == "MSGD" else STOD_FILTER_PATH
    files = []
    prompts = read_json_file(f"./data/dialogues/{datasets}_dataset_final.json")
    file_ls = os.listdir(FILTER_PATH)
    data = []
    bad_response = []
    bad_transition_intent = []
    bad_transition_utter = []
    wrong_intent = []
    id2response = {}
    for f in file_ls:
        if f[-5:] == '.json':
            response = read_json_file(FILTER_PATH+'/'+f)
            id2response[response['id']] = response

    for i, d in enumerate(prompts):
        response = parse_filter(id2response[d['id']]['response'])
        # print(d['id'])
        # print(f"Orignial intent: {d['intent']['type']}")
        # print(id2response[d['id']]['prompt'])
        # print(response)
        if len(response) != 2:
            bad_response.append(d['id'])
            continue
        if "yes" not in response[0].lower().strip():
            bad_transition_intent.append(d['id'])
        if "yes" not in response[1].lower().strip():
            bad_transition_utter.append(d['id'])
    
    print("List of Bad Response")
    print(len(bad_response))
    print(bad_response)
    # print(bad_response)
    print("List of Bad transition (Intent)")
    print(len(bad_transition_intent))
    # print(bad_transition_intent)
    print("List of Bad transition (utterance)")
    print(len(bad_transition_utter))
    # print(bad_transition_utter)
    print("Both bad")
    both = set(bad_transition_intent).intersection(set(bad_transition_utter))
    print(len(both))
    print(len(prompts))
    only_bad_ut = set(bad_transition_utter)-both
    print(len(only_bad_ut))
    print(only_bad_ut)
    intent_cnt_dic = {}
    for d in prompts:
        if d['id'] in only_bad_ut:
            if d['intent']['type'] not in intent_cnt_dic.keys():
                intent_cnt_dic[d['intent']['type']] = 1
            else:
                intent_cnt_dic[d['intent']['type']] += 1
    print(intent_cnt_dic)


    # write_json_file("./bad_id")
    # print(both)


def argparser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', type=str, default='generate', help='generate or rewrite')
    parser.add_argument('--input_file', type=str, help='file of generated prompts')
    parser.add_argument('--datasets', type=str, help='Source datasets')
    parser.add_argument('--output_dir', type=str, help='output directory')
    return parser.parse_args()

if __name__ == '__main__':
    args = argparser()
    if args.mode == 'generate_rewrite':
        print("Generating prompts...")
        generate_rewrite_prompt(args.datasets)
    elif args.mode == 'generate_intent':
        print('Generate intents prompts...')
        generate_intent_det_prompt(args.datasets)
    elif args.mode == "generate_continue":
        print("Generate continue prompt...")
        generate_continue_prompt(args.datasets)
    elif args.mode == "generate_trans":
        print("Generating prompts for transition detection..")
        generate_trans_det_prompt(args.datasets)
    elif args.mode == "generate_dataset":
        print('Generating Dataset..')
        generate_dataset(args.datasets)
    elif args.mode =='rewrite':
        print("Generating rewritten dialogues...")
        generate_from_LLM(args.datasets, args.mode, args.input_file, args.output_dir)
    elif args.mode =="continue":
        print("Generating continued dialogues...")
        generate_from_LLM(args.datasets, args.mode, args.input_file, args.output_dir)
    elif args.mode == "intent_detection":
        print("Detecting Intent...")
        generate_from_LLM(args.datasets, args.mode, args.input_file, args.output_dir)
    elif args.mode == "transition_detection":
        print("Detecting transition")
        generate_from_LLM(args.datasets, args.mode, args.input_file, args.output_dir)
    elif args.mode == "filter":
        print("Filtering the dialogues by LLM")
        LLM_filter(args.datasets,args.input_file,args.output_dir)
    elif args.mode == "filter_out":
        print("Filtering out the dialogues by LLM")
        filter_out(args.datasets)
    elif args.mode == 'print_intent':
        Print_Intent()
    else:
        print(f"No such mode as {args.mode}")






import json
import time
import openai

intent_dic = {
    "LookupMusic":"Search for a song to listen",
    "FindMovie":"Find movies to watch",
    "FindAttraction": "Find attractions to visit",
    "FindBus": "Find bus to take",
    "FindEvents": "Find events to attend to",
    "SearchOnewayFlight": "Search a oneway flight",
    "SearchRoundtripFlights":"Search round-trip flights",
    "SearchHotel": "Search a hotel to stay",
    "GetCarsAvailable": "Search for cars avialable to rent",
    "FindRestaurants": "Find restaurant to go",
    "GetRide":"Find a cab to take"}


MSGD_PATH = './data/dialogues/Merge_SGD.json'
STOD_PATH = './data/dialogues/TOD_Simulators.json'
MSGD_REWRITE_PATH = './final/output_chitchat_only/MSGD'
STOD_REWRITE_PATH = './final/output_chitchat_only/STOD'
MSGD_INTENT_PATH = './final/output_intent/MSGD'
STOD_INTENT_PATH = './final/output_intent/STOD'
MSGD_CONTINUE_PATH = './final/output_continue_one_intent/MSGD'
STOD_CONTINUE_PATH = './final/output_continue_one_intent/STOD'
MSGD_REWRITE_1_PATH = './final/output_chitchat_only_one/MSGD'
STOD_REWRITE_1_PATH = './final/output_chitchat_only_one/STOD'
MSGD_INTENT_1_PATH = './final/output_intent_one/MSGD'
STOD_INTENT_1_PATH = './final/output_intent_one/STOD'
MSGD_CONTINUE_1_PATH = './final/output_continue_one_intent_one/MSGD'
STOD_CONTINUE_1_PATH = './final/output_continue_one_intent_one/STOD'
MSGD_TRANS_PATH = './final/output_transition_detection/MSGD'
STOD_TRANS_PATH = './final/output_transition_detection/STOD'
MSGD_FILTER_PATH = './final/output_filter_why_org/MSGD'
STOD_FILTER_PATH = './final/output_filter/STOD'


CONTINUE_PREFIX = \
"Here is the potential intent(with description) and an incomplete dialogue:\n"

CONTINUE_SUFFIX = \
"Your goal as following: \n \
1. Continue the dialogue with reasonable response considering previous context \n \
2. Continue the dialogue with topics that implicitly related to the intent listed above\n \
3. If you found it hard to transit, please find other topics related to the contexts and intent and chat for several turns before the final transition\n \
4. Continue the topic if it's not yet close to the end\n \
5. For each topic, please generate at least 5 turns\n \
6. The agent should pivot the conversation smoothly, which means the transition invovled longer conversation\n \
7. The user should then somehow mention the given intent, after the pivoted dialogue\n \
8. Use more reasonable phrases to transit the topic of the conversation\n \
9. End the dialogue with Task-oriented Style (TOD) where agent fulfill the user's intent which listed above\n \
\n \
Please note that both the user and agent should not explicitly disclose the intent; instead, the dialogue is naturally guided to the potential purpose.\n \
-----\n \
Output should follow the format below:\n \
\n \
Continued Dialogue:\n \
Agent:.... \n \
User:.....\n \
Agent:....\n \
User:.....\n \
Agent:....\n \
User:.....\n" 

REWRITE_PREFIX = \
"You will be given a conversation between two people.\n \
\n \
 \n \
Here is what you should do:\n \
1. Identify the inconsistent utterances.\n \
2. Give some reasons why they are inconsistent.\n \
3. Modify the dialogue based on previous identified utterances\n \
4. The rewritten dialogue should be more than 6 turns\n \
\n \
Here is the conversation: \n"


REWRITE_SUFFIX = \
"You MUST follow the format as :\n \
Inconsistent utterance:\n \
1.  [utterance]\n \
reason 1:\n \
reason 2:\n \
2. [utterance]\n \
reason 1:\n \
reason 2:\n \
\n \
Rewritten Dialogue:\n \
User:...\n \
Agent:...\n \
User:...\n \
Agent:...\n \
User:...\n \
Agent:...\n \
User:...\n \
"

INTENT_DET_PREFIX = \
"You will be given a dialogue and a list of topics of conversation.\n \
Please tell me which of the following topics will be the most reasonable one to be pivoted to in the dialogue.\n \
\n \
\n \
Here is the dialogue:\n"

INTENT_DET_SUFFIX = \
"Here is the list of topics:\n \
\n \
FindMusic\n \
FindMovie\n \
FindAttraction\n \
FindBus\n \
FindEvents\n \
SearchOnewayFlight\n \
SearchRoundtripFlights\n \
SearchHotel\n \
GetCarsAvailable\n \
FindRestaurants\n \
GetRide\n \
\n \
NOTE:\n \
1. You MUST choose one of the above topic.\n \
2. DONOT create any topics that are not listed above.\n \
3. You should choose the one that is the most related to the topic\n \
The output format should follow the below format:\n \
Potential Topic:\n \
(the one you chose)"

TRANS_DET_PREFIX = \
"You will be given a dialogue below and a potential intent below,"
TRANS_DET_SUFFIX = \
" and your goal is a following:\n \
1. Identify the first utterance that apparently mentions the intent given above\n \
2. You should choose only one turn in the given dialogue\n \
3. The chosen turn should be said by User\n \
\n \
Please follow the output format as below:\n \
\n \
The chosen turn:\n \
..."
def get_response(prompt,task,datasets,dialog,output_dir):
    while True:
        try:
            if task == 'rewrite' or task == 'continue':
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{'role': 'user', 'content': prompt}],
                temperature=1,
                )
            else:
                response = openai.ChatCompletion.create(
                model="gpt-3.5-turbo",
                messages=[{'role': 'user', 'content': prompt}],
                temperature=0,
                )
            break
        except:
            ## sleep if fail
            print("Fail to generate, sleep for 30 seconds")
            time.sleep(30)

    response = response.choices[0]['message']['content']
    dic = {}
    if task == 'Filter':
        dic['id'] = dialog['id']
        dic['response'] = response
        dic['intent'] = dialog['intent']['type']
        dic['prompt'] = dialog['prompt']
        ## write to json
        with open(output_dir+ "/" + datasets + "/" + f"{datasets}_response_" + dialog['id'] + '.json', 'w') as f:
            json.dump(dic, f, indent=4)
        return
    dic['id'] = dialog['id']
    dic['response'] = response
    ## write to json
    with open(output_dir+ "/" + datasets + "/" + f"{datasets}_response_" + dialog['id'] + '.json', 'w') as f:
        json.dump(dic, f, indent=4)
    if task == 'rewrite':
        text = dialog['context']+ "\n" +response
    elif task == 'transition_detection':
        text = response
    elif task == 'intent_detection':
        text = dialog['context'] + '\n' +response
    elif task == 'continue':
        text = dialog['intent']+ "\n"+dialog['context'] + '\n' +response
    write_txt_file(output_dir+ "/"+ datasets +"/" + f"{datasets}_response_"+dialog['id'] + ".txt",text)
    print(f"Finish Task: {task} for {datasets} dialogue: " + dialog['id'])

def read_json_file(path):
    """
    Json Format:
    [{"id": str, "dialogue": [str, str, ...], "intent": {"type":str,"position":int},"transition_candidates":[str,str,...]
    }]"""
    with open(path, 'r') as f:
        data = json.load(f)
    return data
def write_json_file(path, data):
    with open(path, 'w') as f:
        json.dump(data, f, indent=4)
def write_txt_file(path, data):
    with open(path, 'w') as f:
        f.write(data)
def parse_response(response: str):
    response = response.split("\n")
    dialg_flag = 0
    rewritten_dialog = []
    for i, s in enumerate(response):
        s = s.strip()
        if len(s) == 0:
            continue
        if "Rewritten Dialog" in s or "Modified Dialog" in s or "Rewritten dialog" in s or "Modified dialog" in s:
            dialg_flag = 1
            start = s.find(":")
            if len(s[start+1:]) > 0:
                rewritten_dialog.append(s[start+1:])
            continue
        if s == "Inconsistent utterance:":
            dialg_flag = 0
        if dialg_flag == 1:
            rewritten_dialog.append(s)
        else:
            continue
    if "Agent:" in rewritten_dialog[-1] :
        return "\n".join(rewritten_dialog[:-1])
    else:
        return "\n".join(rewritten_dialog)

def parse_intent(response: str):
    response = response.split("\n")
    intent = ""
    description = ""
    for i, s in enumerate(response):
        s = s.strip()
        if "Potential" in s:
            start = s.find(":")
            if len(s[start+1:]) > 0:
                for key in intent_dic.keys():
                    if key in s:
                        intent += key
                        description = intent_dic[key]
                        break
        if i > 0 and "Potential Topic:" in response[i-1]:
            for key in intent_dic.keys():
                if key in s:
                    intent += key
                    description = intent_dic[key]
                    break
            if "FindMusic" in s:
                intent = "LookupMusic"
                description = intent_dic["LookupMusic"]
            if intent == "":
                intent = s.strip()
    intent = intent.strip()
    return intent, description

def parse_continue(response: str):
    response = response.split('\n')
    flag = 0
    continue_dialogue = []
    import re
    for i,s in enumerate(response):
        s = s.strip('\n\t ')
        if "Continued Dialogue" in s:
            flag = 1
            start = s.find(':')
            if len(s[start+1:]) > 0:
                speaker = re.search(r'(User|Agent): (.*)',s).group(1)
                s = re.search(r'(User|Agent): (.*)',s).group(2)
                s = s.strip("\"")
                s = s.strip("\'")
                s = speaker + ":" + " " + s
                continue_dialogue.append(s[start+1:])
        elif flag == 1:
            #TODO:
            # Use regex to parse User: and Agent:
            try:
                speaker = re.search(r'(User|Agent): (.*)',s).group(1)
                s = re.search(r'(User|Agent): (.*)',s).group(2)
                s = s.strip("\"")
                s = s.strip("\'")
                s = speaker + ":" +" " + s
                continue_dialogue.append(s)
            except:
                continue
    if len(continue_dialogue) == 0:
        for i,s in enumerate(response):
            s = s.strip()
            try:
                speaker = re.search(r'(User|Agent): (.*)',s).group(1)
                s = re.search(r'(User|Agent): (.*)',s).group(2)
                s = s.strip("\"")
                s = s.strip("\'")
                s = speaker + ":" + " " + s
                continue_dialogue.append(s)
            except:
                continue

    return "\n".join(continue_dialogue)

def parse_transition(response: str, dialogue):
    response = response.split("\n")
    dialogue = dialogue.split('\n')
    transition = ''
    position = -1
    for s in response:
        s = s.strip("The chosen turn:").strip().strip("\"")
        if s == "":
            continue
        try:
            import re
            s = re.search(r'(User|Agent): (.*)',s).group(2)
            s = s.strip("\"")
            s = s.strip("\'")
            for i, t in enumerate(dialogue):
                if s in t or s.lower() in t.lower():
                    speaker = re.search(r'(User|Agent): (.*)',t).group(1)
                    if speaker == 'User':
                        transition = t
                        position = i
                    elif speaker =='Agent':
                        transition = dialogue[i+1]
                        position = i+1
        except:
            for i, t in enumerate(dialogue):
                if s in t or s.lower() in t.lower():
                    import re
                    speaker = re.search(r'(User|Agent): (.*)',t).group(1)
                    if speaker == 'User':
                        transition = t
                        position = i
                    elif speaker =='Agent':
                        transition = dialogue[i+1]
                        position = i+1
    return transition, position
def parse_filter(response:str):
    response = response.split("\n")
    return response


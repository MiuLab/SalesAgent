# Create Dataset Class for SalesBot 2.0 finetuning
"""
This file contains the Dataset class for SalesBot 2.0 finetuning.
"""

import os
from typing import List, Dict
import json
import torch
from torch.utils.data import Dataset



"""
the json file should be in the following format:
[ { "id": "1",
        "intent" : {
            "type": "intent",
            "description": "intent description",
        },
        "transition_sentence" : {
            "utterance": "transition sentence",
            "position": "1"
        },
        "chitchat_context" : [
            "chitchat context 1",
            "chitchat context 2",
            ...
        ],
        "dialog": [
            "utterance 1",
            "utterance 2",
            ...
        ]
    },
]
"""


    


    
class SalesBot2Dataset(Dataset):
    def __init__(self, data_path, tokenizer, max_len=512):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = self.load_data()
        self.process_data()
    def __len__(self):
        return len(self.datasets)

    def __getitem__(self, index):
        item = self.datasets[index]
        return {
            "dialogue_history": item["dialogue_history"],
            "response": item["response"],
        }

    def load_data(self):
        with open(os.path.join(self.data_path), "r") as f:
            data = json.load(f)
        return data

    def process_data(self):
        # process data for dialogue agent end-to-end finetuning
        self.train_datasets = []
        self.test_datasets = [] 
        id_list = []
        for item in self.data:
            #Gather parts of the dialogue that end with user's turn
            transition_boundary = 1
            intent = ""
            for i in range(len(item["dialog"])):
                # if current turn is transition sentence, response would be <TOD>, and break
                if item["dialog"][i] == item["transition_sentence"]["utterance"]:
                    transition_boundary = i+1
                    break
            chitchat_range = len(item["chitchat_context"]) if transition_boundary > len(item["chitchat_context"]) else 1
            # add intent type to response if the len of dialogue history is equal to chitchat_range
            for i in range(len(item["dialog"])):
                if item["dialog"][i].startswith("User:") and (i+1 >= chitchat_range or i+1 >= chitchat_range-1):
                    intent = item["intent"]["type"]
                    if item["intent"]["type"] == "SearchOnewayFlight" or item["intent"]["type"] == "SearchRoundtripFlights":
                        intent = "SearchFlights"
                    elif item["intent"]["type"] == "GetRide" or item["intent"]["type"] == "FindBus" or item["intent"]["type"] == "GetCarsAvailable":
                        intent = "GetTransportation"

                    # if i + 1 == transition_boundary:
                    #     break
                    # if i+1 == chitchat_range or i+1 == chitchat_range-1:
                    #     item["dialog"][i+1] = f"Agent: The user has implicitly mention the intent of {intent}. I should smoothly pivot the conversation to the topic of {intent}. ## Response: " + item["dialog"][i+1].replace("Agent: ", "")
                    # else:
                    #     item["dialog"][i+1] = f"Agent: The user does not change the topic of {intent}. I should continue the topic. ## Response: " + item["dialog"][i+1].replace("Agent: ","")

                # elif item["dialog"][i].startswith("User:") and i+1 < chitchat_range:
                #     item["dialog"][i+1] = "Agent: The user does not implicitly mention any potential intent, I should continue the chit-chat. ## Response" + item["dialog"][i+1].replace("Agent: ", "")

            # add intent type in the begining of each dialogue history if the len of dialogue history is longer than chitchat
            # for i in range(len(item["dialog"])):
            #     if item["dialog"][i].startswith("User:"):
            #         instruction_template = "Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents']. Think carefully to determine the potential intent and provide suitable response." 
            #         item["dialog"][i] += f" {instruction_template}"



            # make the data into the following format:
            """
            [
                {
                    "id": "<id>_i",
                    "conversation": [
                        {
                            "from": "human",
                            "value": "utterance 1"
                        },
                        {
                            "from": "gpt",
                            "value": "response 1"
                    ]
                }

            ]
            """
            dialogs = []
            for i in range(len(item['dialog'])):
                dic = {}
                dic["id"] = f"{item['id']}_{i}"
                dic["conversations"] = []
                if "User: " in item["dialog"][i]:
                    tmp_1 = {}
                    tmp_2 = {}
                    tmp_1["from"] = "human"
                    tmp_1["value"] = "Dialog History: "
                    # concatenate previous dialog history to the value
                    for j in range(i):
                        if "User: " in item["dialog"][j]:
                            tmp_1["value"] += item["dialog"][j] + "\n"
                        if "Agent: " in item["dialog"][j]:
                            tmp_1["value"] += item["dialog"][j] + "\n"
                    tmp_1["value"] += item["dialog"][i]
                    dic["conversations"].append(tmp_1)
                    if i + 1 ==  transition_boundary:
                        tmp_2["from"] = "gpt"
                        tmp_2["value"] = f"Thought: The user has explicitly shown his/her intent of {intent}.\nResponse: Proceed to task oriented dialog agent"
                        dic["conversations"].append(tmp_2)
                        dialogs.append(dic)
                        break
                    elif i+1 == chitchat_range or i+1 == chitchat_range-1:
                        tmp_2["value"] = f"Thought: The user implicitly mentioned the intent of {intent}. I should smoothly pivot the conversation to the topic of {intent}. \nResponse: " + item["dialog"][i+1].replace("Agent: ", "")
                    elif i+1 > chitchat_range or i+1 > chitchat_range-1:
                        tmp_2["value"] = f"Thought: The user did not change the topic of {intent}. I should continue the topic.\nResponse: " + item["dialog"][i+1].replace("Agent: ","")
                    elif i+1 < chitchat_range:
                        tmp_2["value"] = "Thought: The user did not implicitly mention any potential intent, I should continue the chit-chat.\nResponse: " + item["dialog"][i+1].replace("Agent: ", "")
                    tmp_2["from"] = "gpt"
                    dic["conversations"].append(tmp_2)
                    dialogs.append(dic)
            
            if intent == "SearchFlights": #or intent == "GetTransportation":
                self.test_datasets.extend(dialogs)
            # else:
            #     id_list.append(item["id"])
            #     self.train_datasets.extend(dialogs)
        with open("./data_1213_w_all_CoT/test_CoT_searchflights.json", "w") as f:
            json.dump(self.test_datasets, f, indent=4)
        
        #
        # # save to json_file
        # print(len(self.train_datasets))
        # print(len(self.test_datasets))
        # # random select 500 ids from train to test
        # import random
        # random.shuffle(id_list)
        # test_id_list = id_list[:500]
        # for id in test_id_list:
        #     for i in range(len(self.train_datasets)):
        #         if id in self.train_datasets[i]["id"]:
        #             self.test_datasets.append(self.train_datasets[i])
        # #remove the test data from train
        # for id in test_id_list:
        #     for dic in self.train_datasets:
        #         if id in dic["id"]:
        #             self.train_datasets.remove(dic)
        # # add instructions to the end of "human"
        # for dic in self.train_datasets:
        #     dic["conversations"][0]["value"] += " Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents']. Think carefully to determine the potential intent and provide suitable response given the above dialog history. Output Format: \nThought: <thought>\nResponse: <response>"
        # for dic in self.test_datasets:
        #     dic["conversations"][0]["value"] += " Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents', 'GetTransportation', 'SearchFlights']. Think carefully to determine the potential intent and provide suitable response given the above dialog history. Output Format: \nThought: <thought>\nResponse: <response>"
        #
        # print(len(self.train_datasets))
        # print(len(self.test_datasets))
        # # save to json_file
        # # with open("./data_1024_w_all_intent_prepend/final_data_narrative_in_the_end/train_narrative.json", "w") as f:
        # #     json.dump(self.train_datasets, f, indent=4)
        # # with open("./data_1024_w_all_intent_prepend/final_data_narrative_in_the_end/test.json", "w") as f:
        # #     json.dump(self.test_datasets, f, indent=4)
        # # with open("./data_1024_w_all_intent_prepend/final_data_tod_in_the_end/train_narrative.json", "w") as f:
        # #     json.dump(self.train_datasets, f, indent=4)
        # # with open("./data_1024_w_all_intent_prepend/final_data_tod_in_the_end/test.json", "w") as f:
        # #     json.dump(self.test_datasets, f, indent=4)
        # # make dir
        # import os
        # os.makedirs("./data_1213_w_all_CoT", exist_ok=True)
        # # with open("./data_1027_w_all_CoT/train_CoT.json", "w") as f:
        # #     json.dump(self.train_datasets, f, indent=4)
        # with open("./data_1213_w_all_CoT/test_CoT.json", "w") as f:
        #     json.dump(self.test_datasets, f, indent=4)
        # # with open(os.path.join("./salesbot2_for_llama_finetune.json"), "w") as f:
        # #     json.dump(self.datasets, f, indent=4)
        #             
        #
        #

                



            
    def collate_fn(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        """
        Collate function for the dataset.
        """
        # tokenize the dialogue_history and response in each batch return a tensor of tokenized dialogue_history and response
        # batch is a list of dictionaries, each dictionary contains "dialogue_history", "response"
        self.tokenizer.pad_token = self.tokenizer.eos_token

        for i in range(len(batch)):
            batch[i]["dialogue_history"] = self.tokenizer(batch[i]["dialogue_history"], padding="max_length", truncation=True, max_length=self.max_len)["input_ids"]
            batch[i]["response"] = self.tokenizer(batch[i]["response"], padding="max_length", truncation=True, max_length=self.max_len)["input_ids"]

        # convert batch to the following:
        """
        {
            "input_ids": ..,
            "target_ids": ..
        }
        """
        tokenized_batch = {
            "input_ids": torch.tensor([item["dialogue_history"] for item in batch]),
            "target_ids": torch.tensor([item["response"] for item in batch]),
        }
        return tokenized_batch
        


    



            
                



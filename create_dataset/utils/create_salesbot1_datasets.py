import json
import os

data = json.load(open("salesbot2_prompt/Merge_SGD.json", "r"))
train_data = []
for item in data:
    print(f"Id: {item['id']}")
    dialogs = []
    intent = item["intent"]["type"][0]
    if intent == "PlaySong" or intent == "LookupSong":
        intent = "LookupMusic"
    if intent == "GetTimesForMovie" or intent == "FindMovies":
        intent = "FindMovie"
    chitchat_range = item["intent"]["position"] + 1
    transition_boundary = chitchat_range + 2

    dialog = []
    if chitchat_range % 2 == 0:
        dialog = item["dialog"][1:]
        chitchat_range -= 1
        transition_boundary = chitchat_range + 2
    else:
        dialog = item["dialog"]
    for i in range(len(dialog)):
        print(i)
        dic = {}
        dic["id"] = f"{item['id']}_{i}"
        dic["conversations"] = []
        if i % 2 == 0:
            tmp_1 = {}
            tmp_2 = {}
            tmp_1["from"] = "human"
            tmp_1["value"] = "Dialog History: "
            # concatenate previous dialog history to the value
            for j in range(i + 1):
                if j % 2 == 0:
                    tmp_1["value"] += "User: " + dialog[j] + "\n"
                else:
                    tmp_1["value"] += "Agent: " + dialog[j] + "\n"
            dic["conversations"].append(tmp_1)
            if i + 1 == transition_boundary:
                tmp_2["from"] = "gpt"
                tmp_2["value"] = (
                    f"Thought: The user has explicitly shown his/her intent of {intent}.\nResponse Proceed to task oriented dialog agent"
                )
                dic["conversations"].append(tmp_2)
                dialogs.append(dic)
                break
            elif i + 1 == chitchat_range:
                tmp_2["value"] = (
                    f"Thought: The user implicitly mentioned the intent of {intent}. I should smoothly pivot the conversation to the topic of {intent}. \nResponse: "
                    + dialog[i + 1].replace("Agent: ", "")
                )
            elif i + 1 < chitchat_range:
                tmp_2["value"] = (
                    "Thought: The user did not implicitly mention any potential intent, I should continue the chit-chat.\nResponse: "
                    + dialog[i + 1].replace("Agent: ", "")
                )
            tmp_2["from"] = "gpt"
            dic["conversations"].append(tmp_2)
            dialogs.append(dic)
    train_data.extend(dialogs)

for dic in train_data:
    dic["conversations"][0]["value"] += (
        " Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindMovie', 'LookUpMusic']. Think carefully to determine the potential intent and provide suitable response given the above dialog history. Output Format: \nThought: <thought>\nResponse: <response>"
    )

# write to file
os.makedirs("./data_baseline_salesbot1/", exist_ok=True)
with open("./data_baseline_salesbot1/train_CoT.json", "w") as outfile:
    json.dump(train_data, outfile, indent=4)

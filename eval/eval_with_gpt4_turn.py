import os
import tqdm
import argparse
import json
import time
import openai
openai.api_key = "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"    # Azure 的密鑰
openai.api_version = "2023-09-01-preview" # API 版本，未來可能會變
model = "gpt-4"  # 模型的部署名
temperature = 0

CONTEXT = """
The following is a conversation between a user and a salesbot, and
the goal of salesbot is to smoothly direct the conversation toward a certain topic and proceed to task-oriented dialogue agent.

"""
EVAL_SCHEMA = """
Definition of the scores:
- Naturalness: The content of the dialogue is in general natural and human-like.
- Consistancy: The dialogue is coherent and consistent.
{
    "naturalness": {
        "reason": "<reason for naturalness score>",
        "score": <naturalness score>
        },
    "consistancy": {
        "reason": "<reason for coherence score>",
        "score": <coherence score>
        }
}
"""
TEMPLATE = """
{context}
Score the following dialogue generated on a continuous scale from 0 to 100.
Dialogue: {dialog}
Format:
{eval_schema}
Output:
"""
def format_dialog(dialog):
    """
    Dialog Dict:
        [
                {
                    "role": "user",
                    "content": "<content>"
                    },
                {
                    "role": "assistant",
                    "content": "<content>"
                    },
                ...
        ]
    Format:
        User: <content>
        Agent: <content>
        ...
    """
    result = ""
    for d in dialog[2:]:
        if d["role"] == "user":
            result += "User: " + d["content"] + "\n"
        elif d["role"] == "assistant":
            result += "Agent: " + d["content"] + "\n"
    return result
def do_eval(dialog):

    prompt = TEMPLATE.format(context=CONTEXT,eval_schema=EVAL_SCHEMA,dialog=dialog)
    response = openai.ChatCompletion.create(
        engine=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response

def main(args):
    with open(args.input, "r") as f:
        data = json.load(f)

    data_with_last_turn = []
    prev_id_prefix = ""
    for i, d in enumerate(data):
        id_prefix = "_".join(d["id"].split('_')[:2])
        if (id_prefix != prev_id_prefix and i != 0) or i == len(data) - 1:
            conv = data[i-1]["conversations"]
            history = conv[0]["value"].split(" Here is a list of potential intents")[0].split("Dialog History: ")[-1]
            dialog = history + "\nAgent: " + conv[1]["value"].split("Response: ")[-1]
            data[i-1]["dialog"] = dialog
            data_with_last_turn.append(data[i-1])
        prev_id_prefix = id_prefix
    progress_bar = tqdm.tqdm(total=len(data_with_last_turn))
    for d in data_with_last_turn:
        conv = d["dialog"]
        d["score"] = []
        progress_bar.update(1)
        response = do_eval(conv)
        # print(response["choices"][0]["message"]["content"])
        d["score"].append(json.loads(response["choices"][0]["message"]["content"]))
       # sleep for 1 second to avoid rate limit 
        time.sleep(1)
    results = data_with_last_turn
    with open(args.output, "w") as f:
        json.dump(results, f, indent=2)
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()
if __name__ == "__main__":
    args = arg_parser()
    main(args)

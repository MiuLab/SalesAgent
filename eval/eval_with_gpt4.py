import os
import tqdm
import argparse
import json
import time
import openai
openai.api_key = "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"    # Azure 的密鑰
model = "gpt-4"  # 模型的部署名
temperature = 0

CONTEXT = """
The following is a conversation between a user and a salesbot, and
the goal of salesbot is to smoothly direct the conversation toward a certain topic and proceed to task-oriented dialogue agent.

"""
EVAL_SCHEMA = """
Definition of the scores:
- Naturalness (the higher the more natural): The content of the dialogue is in general natural and human-like.
- Coherence (ther higher the more coherent): The dialogue is coherent and easy to follow.
- Smoothness (the higher the smoother): The dialogue is smooth and the agent is able to smoothly steer the conversation toward a certain topic without explicit transition.
- Agent aggressiveness (the higher the more aggressive): The agent is considered to be aggressive if it is too pushy and it change the topic without the user's intention. Especially, if it directly proceed to task-oriented dailogue agent or directly provide user intents options without the user's explicit intent, the agent is very aggressive.
- Agent consistancy (the higher the more consistant): The agent is considered as consistant if it is able to maintain the same personality and topic throughout the conversation.
{
    "naturalness": {
        "reason": "<reason for naturalness score>",
        "score": <naturalness score>
        },
    "coherence": {
        "reason": "<reason for coherence score>",
        "score": <coherence score>
        },
    "smoothness": {
        "reason": "<reason for smoothness score>",
        "score": <smoothness score>
        },
    "agent aggressiveness": {
        "reason": "<reason for agent aggressiveness score>",
        "score": <agent aggressiveness score>
        },
    "agent consistancy": {
        "reason": "<reason for agent consistancy score>",
        "score": <agent consistancy score>
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
            result += "Agent: " + d["content"] + "\n"
        elif d["role"] == "assistant":
            result += "User: " + d["content"] + "\n"
    return result
def do_eval(dialog):

    dialog = format_dialog(dialog)
    prompt = TEMPLATE.format(context=CONTEXT,eval_schema=EVAL_SCHEMA,dialog=dialog)
    response = openai.ChatCompletion.create(
        model=model,
        temperature=temperature,
        messages=[{"role": "user", "content": prompt}],
    )
    return response

def main(args):
    with open(args.input, "r") as f:
        data = json.load(f)

    progress_bar = tqdm.tqdm(total=250)
    for d in data:
        conv = d["conversations"]
        d["score"] = []
        for k, v in conv.items():
            progress_bar.update(1)
            response = do_eval(v)
            d["score"].append(json.loads(response["choices"][0]["message"]["content"]))
       # sleep for 1 second to avoid rate limit 
        time.sleep(0.5)
        # log
        with open(args.output, "w") as f:
            json.dump(data, f, indent=2)
def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    return parser.parse_args()
if __name__ == "__main__":
    args = arg_parser()
    main(args)

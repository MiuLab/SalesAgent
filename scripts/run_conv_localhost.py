import os
import argparse

import torch

from fastchat.model import load_model, get_conversation_template, add_model_args

from tqdm.auto import tqdm
import json
import copy
import openai

AGNET_PREFIX = "Dialogue History:"
AGENT_SUFFIX = "Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents']. Think carefully to determine the potential intent and provide suitable response given the above dialog history. Output Format: \nThought: <thought>\nResponse: <response>"
AGENT_SUFFIX_LLAMA = "Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents']. Think carefully to determine the potential intent and provide suitable response given the above dialog history. You should response as a real conversation.\n If you think user has explicitly mentioned the above intent, you should say \"Proceed to task oriented dialog agent.\""
USER_SUFFIX = "Imagine you are a real person. You are having chat with a online agent, so the repsonse do not include any expresssions. Remember, maintain a natural tone. Your response should be only your text resposne without any other expressions. Keep it as short as possible.\n"
USER_SUFFIX_NEG_ALL = "You are not interested in FindAttraction, FindRestaurants, FindMovie, LookUpMusic, SearchHotel, FindEvents, if the agent ask any about one of them, donot ask for any recommendations and you should say, \"I don't want to talk about this. Let's talk about something else\". Note that you should be more firm."
USER_SUFFIX_NEG = "You are not interested in {intents}, if the agent ask any about one of them, donot ask for any recommendations and you should say, \"I don't want to talk about this. Let's talk about something else\". Note that you should be more firm."
# USER_SUFFIX_NEG = "You are not interested in 'FindAttraction', 'FindRestaurants'. Do not continue these topics."


@torch.inference_mode()
def get_user_reponse(history):
    # to get proper authentication, make sure to use a valid key that's listed in
    # the --api-keys flag. if no flag value is provided, the `api_key` will be ignored.

    # TODO: The 'openai.api_base' option isn't read in the client API. You will need to pass it when you instantiate the client, e.g. 'OpenAI(api_base="http://140.112.29.239:5000/")'
    # openai.api_base = "http://140.112.29.239:5000/"

    model = "Llama-2-7b-chat-hf"

    # create message from history
    # create a chat completion

    openai.api_key = "EMPTY"
    openai.api_base = "http://localhost:5000/v1"
    completion = openai.ChatCompletion.create(
        model=model,
        messages=history,
        top_p=args.user_topp,
        temperature=args.user_temperature,
        stop=["\n"],
    )
    # print the completion
    response = completion.choices[0].message.content
    return response


def main(args):
    # Load model
    agent_model, agent_tokenizer = load_model(
        args.agent_model,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    # load persona from persona.json
    # if output file exitst, load it and continue
    if os.path.exists(args.output_file):
        with open(args.output_file, "r") as f:
            personas = json.load(f)
    else:
        with open("persona.json", "r") as f:
            personas = json.load(f)
    # Build the prompt with a conversation template
    with open("persona_with_conv.json", "r") as f:
        ref = json.load(f)

    conv_cnt = 0
    # neg_labels = ["no_preference", "not_interested_2", "not_interested_4", "not_interested_all"]
    # give me a list of 250 long with 0,1,2,3 randomly equalled distributed
    # label_ls = []
    # for i in range(250):
    #     label_ls.append(neg_labels[i%4])

    # random.seed(0)
    # label_ls = random.sample(label_ls, 250)

    for i, persona in enumerate(tqdm(personas)):
        if "conversations" in persona and len(persona["conversations"]) == 5:
            continue
        print(f"Persona: {persona['persona']}")
        persona["conversations"] = {}
        persona["negativeness"] = ref[i]["negativeness"]
        persona["not_interested_in"] = ref[i]["not_interested_in"]
        persona["terminate_reason"] = []
        persona["num_turns"] = []
        for _ in range(5):
            num_turn = 0
            history = []
            history.extend(
                [
                    {
                        "role": "system",
                        "content": "You are not an AI. " + persona["persona"],
                    },
                    {"role": "user", "content": "Hi."},
                ]
            )
            # neg_label = label_ls[conv_cnt]
            neg_label = persona["negativeness"][_]
            conv_cnt += 1
            print(f"Negativeness: {neg_label}")
            if neg_label == "no_preference":
                history[0]["content"] += "\n" + USER_SUFFIX
                # persona["not_interested_in"].append("None")
            elif neg_label == "not_interested_2":
                # random.seed(conv_cnt)
                # intents = ", ".join(random.sample(['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents'], 2))
                intents = persona["not_interested_in"][_]
                history[0]["content"] += (
                    " "
                    + USER_SUFFIX_NEG.replace("{intents}", intents)
                    + "\n"
                    + USER_SUFFIX
                )
                # persona["not_interested_in"].append(intents)
            elif neg_label == "not_interested_4":
                # random.seed(conv_cnt)
                # intents = ", ".join(random.sample(['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents'], 4))
                intents = persona["not_interested_in"][_]
                history[0]["content"] += (
                    " "
                    + USER_SUFFIX_NEG.replace("{intents}", intents)
                    + "\n"
                    + USER_SUFFIX
                )
                # persona["not_interested_in"].append(intents)
            elif neg_label == "not_interested_all":
                history[0]["content"] += " " + USER_SUFFIX_NEG_ALL + "\n" + USER_SUFFIX
                # persona["not_interested_in"].append("FindAttraction, FindRestaurants, FindMovie, LookUpMusic, SearchHotel, FindEvents")

            # persona["negativeness"].append(neg_label)
            print(history)
            while True:
                # if random.random() < 0.5:
                msg = get_user_reponse(history)
                # else:
                #     msg = "I don't want to talk about this. Let's talk about something else."
                history.append(
                    {
                        "role": "assistant",
                        "content": msg,
                    }
                )
                print(f"User: {msg}")
                history_string = ""
                if args.agent_model != "meta-llama/Llama-2-7b-chat-hf":
                    for turn in history[1:-1]:
                        role = ""
                        if turn["role"] == "assistant":
                            role = "User"
                        else:
                            role = "Agent"
                        history_string += role + ": " + turn["content"] + "\n"
                else:
                    for turn in history[1:]:
                        role = ""
                        if turn["role"] == "assistant":
                            role = "User"
                        else:
                            role = "Agent"
                        history_string += role + ": " + turn["content"] + "\n"
                msg_prompt = ""
                if args.agent_model != "meta-llama/Llama-2-7b-chat-hf":
                    msg_prompt = AGNET_PREFIX + history_string + AGENT_SUFFIX
                else:
                    msg_prompt = AGNET_PREFIX + history_string + AGENT_SUFFIX_LLAMA
                conv = get_conversation_template(args.model_path)
                if args.agent_model == "meta-llama/Llama-2-7b-chat-hf":
                    conv.set_system_message(msg_prompt)
                    # print(conv.name)
                    conv.append_message(conv.roles[0], msg)
                    conv.append_message(conv.roles[1], None)
                else:
                    conv.append_message(conv.roles[0], msg_prompt)
                    conv.append_message(conv.roles[1], None)

                prompt = conv.get_prompt()

                # Run inference
                inputs = agent_tokenizer([prompt], return_tensors="pt").to(args.device)
                print(f"Token len: {len(inputs['input_ids'][0])}")
                if len(inputs["input_ids"][0]) > 2048:
                    print("Input length exceeds 2048, break")
                    persona["terminate_reason"].append("Input length exceeds 2048")
                    persona["num_turns"].append(num_turn)
                    break
                # if outputs does not include "Response" generate til it has
                while True:
                    output_ids = agent_model.generate(
                        **inputs,
                        do_sample=True if args.agent_temperature > 1e-5 else False,
                        temperature=args.agent_temperature,
                        repetition_penalty=args.agent_repetition_penalty,
                        max_new_tokens=args.agent_max_new_tokens,
                    )
                    if agent_model.config.is_encoder_decoder:
                        output_ids = output_ids[0]
                    else:
                        output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
                    outputs = agent_tokenizer.decode(
                        output_ids,
                        skip_special_tokens=True,
                        spaces_between_special_tokens=False,
                    )
                    if args.agent_model != "meta-llama/Llama-2-7b-chat-hf":
                        # if "Response" occur 1 time in outputs break
                        if outputs.count("Response") == 1:
                            break
                    else:
                        break

                thought = ""
                print(outputs)
                if args.agent_model != "meta-llama/Llama-2-7b-chat-hf":
                    thought, outputs = outputs.split("Response")
                else:
                    outputs = outputs.split("USER")[0].split("User")[0].strip()
                outputs = outputs.strip(": ")
                print(thought.strip())
                print(f"Agent: {outputs}")
                history.append(
                    {
                        "role": "user",
                        "content": outputs,
                        "thought": thought.strip(),
                    }
                )
                num_turn += 2
                if "Proceed to task oriented dialog agent" in outputs:
                    persona["terminate_reason"].append("Success")
                    persona["num_turns"].append(num_turn)
                    break
                if (
                    "bye" in outputs.lower()
                    or "goodbye" in outputs.lower()
                    or "good bye" in outputs.lower()
                ):
                    persona["terminate_reason"].append("Conversation End")
                    persona["num_turns"].append(num_turn)
                    break
                if num_turn == args.max_turns:
                    print("Reach max turns: break")
                    persona["terminate_reason"].append("Reach Max Turns")
                    persona["num_turns"].append(num_turn)
                    break
            persona["conversations"][f"conv_{_}"] = copy.deepcopy(history)
            torch.cuda.empty_cache()
        # clear cuda memory

        with open(args.output_file, "w") as f:
            json.dump(personas, f, indent=4)


def arg_parser():
    # agent's param
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument(
        "--agent_model",
        type=str,
        default="morris-chang/SalesBot2_CoT_Lora_add_thought",
        help="agents model",
    )
    parser.add_argument(
        "--agent_temperature", type=float, default=0.9, help="agents temperature"
    )
    parser.add_argument("--agent_topk", type=int, default=50, help="agents topk")
    parser.add_argument("--agent_topp", type=float, default=1, help="agents topp")
    parser.add_argument(
        "--agent_max_new_tokens", type=int, default=200, help="agents max new tokens"
    )
    parser.add_argument(
        "--agent_repetition_penalty",
        type=float,
        default=1.0,
        help="agents repetition penalty",
    )
    parser.add_argument(
        "--agent_do_sample", type=bool, default=True, help="agents no sample"
    )
    # user's param
    parser.add_argument("--user_model", type=str, default=None, help="users model")
    parser.add_argument(
        "--user_temperature", type=float, default=0.5, help="users temperature"
    )
    parser.add_argument("--user_topk", type=int, default=50, help="users topk")
    parser.add_argument("--user_topp", type=float, default=1, help="users topp")
    parser.add_argument(
        "--user_max_new_tokens", type=int, default=100, help="users max new tokens"
    )
    parser.add_argument(
        "--user_repetition_penalty",
        type=float,
        default=1.0,
        help="users repetition penalty",
    )
    parser.add_argument(
        "--user_do_sample", type=bool, default=True, help="users no sample"
    )
    # common param
    parser.add_argument("--max_turns", type=int, default=20, help="max turns")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--output_file", type=str, default="persona_with_conv.json")

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = arg_parser()
    main(args)

"""
Use FastChat with Hugging Face generation APIs.

Usage:
python3 -m fastchat.serve.huggingface_api --model lmsys/vicuna-7b-v1.5
python3 -m fastchat.serve.huggingface_api --model lmsys/fastchat-t5-3b-v1.0
"""
import os
import argparse

import torch

from fastchat.model import load_model, get_conversation_template, add_model_args

from tqdm.auto import tqdm
import json
import copy
PREFIX = "Dialogue History:"
SUFFIX = "Here is a list of potential intents that might be referred by the user: ['FindAttraction', 'FindRestaurants', 'FindMovie', 'LookUpMusic', 'SearchHotel', 'FindEvents', 'GetTransportation', 'SearchFlights']. Think carefully to determine the potential intent and provide suitable response given the above dialog history. Output Format: \nThought: <thought>\nResponse: <response>"
@torch.inference_mode()
def main(args):
    # Load model
    model, tokenizer = load_model(
        args.model_path,
        device=args.device,
        num_gpus=args.num_gpus,
        max_gpu_memory=args.max_gpu_memory,
        load_8bit=args.load_8bit,
        cpu_offloading=args.cpu_offloading,
        revision=args.revision,
        debug=args.debug,
    )

    # Build the prompt with a conversation template
        #Load data and get data message
    with open(args.test_data_path, "r") as f:
        test_dataset = json.load(f)
    
    outputs = []
    for item in tqdm(test_dataset):

        dialog = item["conversations"]
        msg = dialog[0]["value"]
        conv = get_conversation_template(args.model_path)
        item['outputs'] = []
        conv.append_message(conv.roles[0], msg)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()

        # Run inference
        inputs = tokenizer([prompt], return_tensors="pt").to(args.device)
        output_ids = model.generate(
            **inputs,
            do_sample=True if args.temperature > 1e-5 else False,
            temperature=args.temperature,
            repetition_penalty=args.repetition_penalty,
            max_new_tokens=args.max_new_tokens,
        )

        if model.config.is_encoder_decoder:
            output_ids = output_ids[0]
        else:
            output_ids = output_ids[0][len(inputs["input_ids"][0]) :]
        outputs = tokenizer.decode(
            output_ids, skip_special_tokens=True, spaces_between_special_tokens=False
        )
        dic = {"from": conv.roles[1], "value": outputs}

        # Print results
        # print(history)
        # print(f"{conv.roles[0]}: {msg}")
        # print(f"{conv.roles[1]}: {outputs}")
        item["response"] = dic
    # save to output.json
    with open(os.path.join(args.output_path, "output.json"), "w") as f:
        json.dump(test_dataset, f, indent=4)

                


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_model_args(parser)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--repetition_penalty", type=float, default=1.0)
    parser.add_argument("--max-new-tokens", type=int, default=100)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--test_data_path", type=str, default="./data/final_data_narrative_in_the_end/test.json")
    parser.add_argument("--output_path", type=str, default="./")
    # parser.add_argument("--message", type=str, default="Hello! Who are you?")


    args = parser.parse_args()
    # check if output_dir exists
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)


    # Reset default repetition penalty for T5 models.
    if "t5" in args.model_path and args.repetition_penalty == 1.0:
        args.repetition_penalty = 1.2

    main(args)
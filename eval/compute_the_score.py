import json
import os
import argparse

"""
Here is the format of json file
[
        {
            "id": "0",
            "persona": "<persona>",
            "conversations":{
                "<num_of_turns>": [
                {
                    "role": "<role>",
                    "content": "<content>"
                },
                ...
                ]
            },
            "negativeness": [<negativeness_1>, <negativeness_2>, <negativeness_3>, <negativeness_4>, <negativeness_5>],
            "not_interested": [<not_interested_1>, <not_interested_2>, <not_interested_3>, <not_interested_4>, <not_interested_5>],
            "terminate_reason": [<terminate_reason_1>, <terminate_reason_2>, <terminate_reason_3>, <terminate_reason_4>, <terminate_reason_5>],
            "score": [<score_1>, <score_2>, <score_3>, <score_4>, <score_5>] # each of them is a dictionary with naturalness, coherence, agent aggresiveness, smoothness and agent consistancy
        ]
"""
def main(args):
    # READ FILE
    neg_labels = ["no_preference", "not_interested_2", "not_interested_4", "not_interested_all"]
    terminate_reasons = ["no_preference", "not_interested_2", "not_interested_4", "not_interested_all"]
    with open(args.model_eval, 'r') as f:
        model_eval = json.load(f)
    with open(args.baseline_eval, 'r') as f:
        baseline_eval = json.load(f)
    with open(args.llama_baseline_eval, 'r') as f:
        llama_baseline_eval = json.load(f)
    # there is an typo of aggresiveness, and consistency in the file, so I change it to aggressiveness, consistancy
    for i in range(len(model_eval)):
        score = model_eval[i]['score']
        for i ,score_dict in enumerate(score):
            try:
                score_dict['agent aggressiveness'] = score_dict.pop('agent aggresiveness')
            except:
                pass
            try:
                score_dict['agent consistancy'] = score_dict.pop('agent consistency')
            except:
                pass

    for i in range(len(baseline_eval)):
        score = baseline_eval[i]['score']
        for i ,score_dict in enumerate(score):
            try:
                score_dict['agent aggressiveness'] = score_dict.pop('agent aggresiveness')
            except:
                pass
            try:
                score_dict['agent consistancy'] = score_dict.pop('agent consistency')
            except:
                pass
    for i in range(len(llama_baseline_eval)):
        score = llama_baseline_eval[i]['score']
        for i ,score_dict in enumerate(score):
            try:
                score_dict['agent aggressiveness'] = score_dict.pop('agent aggresiveness')
            except:
                pass
            try:
                score_dict['agent consistancy'] = score_dict.pop('agent consistency')
            except:
                pass

    # compute number of label for two models
    model_label_count = {"no_preference": 0, "not_interested_2": 0, "not_interested_4": 0, "not_interested_all": 0}
    baseline_label_count = {"no_preference": 0, "not_interested_2": 0, "not_interested_4": 0, "not_interested_all": 0}
    for i in range(len(model_eval)):
        negativeness = model_eval[i]['negativeness']
        conv = model_eval[i]['conversations']
        for i in range(len(conv)):
            label = negativeness[i]
            model_label_count[label] += 1
    for i in range(len(baseline_eval)):
        negativeness = baseline_eval[i]['negativeness']
        conv = baseline_eval[i]['conversations']
        for i in range(len(conv)):
            label = negativeness[i]
            baseline_label_count[label] += 1
    # Compute success  rate
    model_success = {"no_preference": 0, "not_interested_2": 0, "not_interested_4": 0, "not_interested_all": 0}
    model_success_avg_turns = {"no_preference": 0, "not_interested_2": 0, "not_interested_4": 0, "not_interested_all": 0}
    baseline_success = {"no_preference": 0, "not_interested_2": 0, "not_interested_4": 0, "not_interested_all": 0}
    baseline_success_avg_turns = {"no_preference": 0, "not_interested_2": 0, "not_interested_4": 0, "not_interested_all": 0}
    llama_baseline_success = {"no_preference": 0, "not_interested_2": 0, "not_interested_4": 0, "not_interested_all": 0}
    llama_baseline_success_avg_turns = {"no_preference": 0, "not_interested_2": 0, "not_interested_4": 0, "not_interested_all": 0}
    for i in range(len(model_eval)):
        terminate_reason = model_eval[i]['terminate_reason']
        negativeness = model_eval[i]['negativeness']
        num_of_turns = model_eval[i]['num_turns']
        for i ,reason in enumerate(terminate_reason):
            # if reason is "Success"
            if reason == "Success":
                model_success[negativeness[i]] += 1
                model_success_avg_turns[negativeness[i]] += num_of_turns[i]
    for i in range(len(baseline_eval)):
        terminate_reason = baseline_eval[i]['terminate_reason']
        negativeness = baseline_eval[i]['negativeness']
        num_of_turns = baseline_eval[i]['num_turns']
        for i ,reason in enumerate(terminate_reason):
            # if reason is "Success"
            if reason == "Success":
                baseline_success[negativeness[i]] += 1
                baseline_success_avg_turns[negativeness[i]] += num_of_turns[i]

    for i in range(len(llama_baseline_eval)):
        terminate_reason = llama_baseline_eval[i]['terminate_reason']
        negativeness = llama_baseline_eval[i]['negativeness']
        num_of_turns = llama_baseline_eval[i]['num_turns']
        for i ,reason in enumerate(terminate_reason):
            # if reason is "Success"
            if reason == "Success":
                llama_baseline_success[negativeness[i]] += 1
                llama_baseline_success_avg_turns[negativeness[i]] += num_of_turns[i]
    #print success rate for each label and overall
    print("Model Success Rate")
    for label in neg_labels:
        # print beautiful
        print(f"    Negativeness: {label}")
        print(f"    Success Rate: {model_success[label]/model_label_count[label]}")
    print(" Overall Success Rate: ", sum(model_success.values())/sum(model_label_count.values()))
    print(" Overall Average Turns: ", sum(model_success_avg_turns.values())/sum(model_success.values()))

    print("Baseline Success Rate")
    for label in neg_labels:
        # print beautiful
        print(f"    Negativeness: {label}")
        print(f"    Success Rate: {baseline_success[label]/baseline_label_count[label]}")
    print(" Overall Success Rate: ", sum(baseline_success.values())/sum(baseline_label_count.values()))
    print(" Overall Average Turns: ", sum(baseline_success_avg_turns.values())/sum(baseline_success.values()))

    print("LLAMA Baseline Success Rate")
    for label in neg_labels:
    # print beautiful
        print(f"    Negativeness: {label}")
        print(f"    Success Rate: {llama_baseline_success[label]/baseline_label_count[label]}")
    print(" Overall Success Rate: ", sum(llama_baseline_success.values())/sum(baseline_label_count.values()))
    print(" Overall Average Turns: ", sum(llama_baseline_success_avg_turns.values())/sum(llama_baseline_success.values()))
    

    # Compute average score, for naturalness, coherence, agent aggresiveness, smoothness and agent consistancy, for each label
    score_label = ["naturalness", "coherence", "agent aggressiveness", "smoothness", "agent consistency"]
    model_avg_score = {"no_preference": [0, 0, 0, 0, 0], "not_interested_2": [0, 0, 0, 0, 0], "not_interested_4": [0, 0, 0, 0, 0], "not_interested_all": [0, 0, 0, 0, 0]}
    baseline_avg_score = {"no_preference": [0, 0, 0, 0, 0], "not_interested_2": [0, 0, 0, 0, 0], "not_interested_4": [0, 0, 0, 0, 0], "not_interested_all": [0, 0, 0, 0, 0]}
    llama_baseline_avg_score = {"no_preference": [0, 0, 0, 0, 0], "not_interested_2": [0, 0, 0, 0, 0], "not_interested_4": [0, 0, 0, 0, 0], "not_interested_all": [0, 0, 0, 0, 0]}
    for i in range(len(model_eval)):
        score = model_eval[i]['score']
        negativeness = model_eval[i]['negativeness']
        for i ,score_dict in enumerate(score):
            # if reason is "Success"
            model_avg_score[negativeness[i]][0] += int(score_dict['naturalness']['score'])
            model_avg_score[negativeness[i]][1] += int(score_dict['coherence']['score'])
            model_avg_score[negativeness[i]][2] += int(score_dict['agent aggressiveness']['score'])
            model_avg_score[negativeness[i]][3] += int(score_dict['smoothness']['score'])
            model_avg_score[negativeness[i]][4] += int(score_dict['agent consistancy']['score'])

    for i in range(len(baseline_eval)):
        score = baseline_eval[i]['score']
        negativeness = baseline_eval[i]['negativeness']
        for i ,score_dict in enumerate(score):
            # if reason is "Success"
            baseline_avg_score[negativeness[i]][0] += int(score_dict['naturalness']['score'])
            baseline_avg_score[negativeness[i]][1] += int(score_dict['coherence']['score'])
            baseline_avg_score[negativeness[i]][2] += int(score_dict['agent aggressiveness']['score'])
            baseline_avg_score[negativeness[i]][3] += int(score_dict['smoothness']['score'])
            baseline_avg_score[negativeness[i]][4] += int(score_dict['agent consistancy']['score'])
    for i in range(len(llama_baseline_eval)):
        score = llama_baseline_eval[i]['score']
        negativeness = llama_baseline_eval[i]['negativeness']
        for i ,score_dict in enumerate(score):
            # if reason is "Success"
            llama_baseline_avg_score[negativeness[i]][0] += int(score_dict['naturalness']['score'])
            llama_baseline_avg_score[negativeness[i]][1] += int(score_dict['coherence']['score'])
            llama_baseline_avg_score[negativeness[i]][2] += int(score_dict['agent aggressiveness']['score'])
            llama_baseline_avg_score[negativeness[i]][3] += int(score_dict['smoothness']['score'])
            llama_baseline_avg_score[negativeness[i]][4] += int(score_dict['agent consistancy']['score'])
#print average score for each label and overall
    
    print("Model Average Score")
    for label in neg_labels:
        # print beautiful
        print(f"    Negativenesls: {label}")
        for s_label, score in zip(score_label, model_avg_score[label]):
            print(f"        {s_label}: {score/model_label_count[label]}")
    print("Overall Average Score")
    sum_of_model_score_= [0,0,0,0,0]
    for score_ls in model_avg_score.values():
        sum_of_model_score_[0] += score_ls[0]
        sum_of_model_score_[1] += score_ls[1]
        sum_of_model_score_[2] += score_ls[2]
        sum_of_model_score_[3] += score_ls[3]
        sum_of_model_score_[4] += score_ls[4]
    for s_label, score in zip(score_label, sum_of_model_score_):
        print(f"        {s_label}: {score/sum(model_label_count.values())}")
    print("Baseline Average Score")
    for label in neg_labels:
        # print beautiful
        print(f"    Negativenesls: {label}")
        for s_label, score in zip(score_label, baseline_avg_score[label]):
            print(f"        {s_label}: {score/baseline_label_count[label]}")
    print("Overall Average Score")
    sum_of_baseline_score = [0,0,0,0,0]
    for score_ls in baseline_avg_score.values():
        sum_of_baseline_score[0] += score_ls[0]
        sum_of_baseline_score[1] += score_ls[1]
        sum_of_baseline_score[2] += score_ls[2]
        sum_of_baseline_score[3] += score_ls[3]
        sum_of_baseline_score[4] += score_ls[4]
    for s_label, score in zip(score_label, sum_of_baseline_score):
        print(f"        {s_label}: {score/sum(baseline_label_count.values())}")
    print("LLAMA Baseline Average Score")
    for label in neg_labels:
    # print beautiful
        print(f"    Negativenesls: {label}")
        for s_label, score in zip(score_label, llama_baseline_avg_score[label]):
            print(f"        {s_label}: {score/baseline_label_count[label]}")
    print("Overall Average Score")
    sum_of_llama_baseline_score = [0,0,0,0,0]
    for score_ls in llama_baseline_avg_score.values():
        sum_of_llama_baseline_score[0] += score_ls[0]
        sum_of_llama_baseline_score[1] += score_ls[1]
        sum_of_llama_baseline_score[2] += score_ls[2]
        sum_of_llama_baseline_score[3] += score_ls[3]
        sum_of_llama_baseline_score[4] += score_ls[4]
            
    for s_label, score in zip(score_label, sum_of_llama_baseline_score):
        print(f"        {s_label}: {score/sum(baseline_label_count.values())}")


    # print number of label for each label

    print("Model Number of Label")
    for label in neg_labels:
        # print beautiful
        print(f"    Negativeness: {label}")
        print(f"    Number of Label: {model_label_count[label]}")

    print("Overall Number of Label: ", sum(model_label_count.values()))
    print("Baseline Number of Label")
    for label in neg_labels:
        # print beautiful
        print(f"    Negativeness: {label}")
        print(f"    Number of Label: {baseline_label_count[label]}")

    print("Overall Number of Label: ", sum(baseline_label_count.values()))
    print("LLAMA Baseline Number of Label")
    for label in neg_labels:
        # print beautiful
        print(f"    Negativeness: {label}")
        print(f"    Number of Label: {baseline_label_count[label]}")
    print("Overall Number of Label: ", sum(baseline_label_count.values()))


def args_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_eval', type=str, default='./persona_with_conv_with_eval.json')
    parser.add_argument('--baseline_eval', type=str, default='./persona_with_conv_salesbot1_baseline_with_eval.json')
    parser.add_argument('--llama_baseline_eval', type=str, default='./persona_with_conv_llama_baseline_with_eval.json')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
    args = args_parser()
    main(args)

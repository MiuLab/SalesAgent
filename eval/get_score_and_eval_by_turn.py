import json

def main():
    # load data score
    """
    [
            {
                "score":{
                    "naturalness":{
                        "score": "<score>",
                        "reason": "<reason>"
                        }
                    "consistancy":{
                        "score": "<score>",
                        "reason": "<reason>"
                        }
                    }
            },
            {
            "score":{
                "naturalness":{
                    "score": "<score>",
                    "reason": "<reason>"
                    }
                "consistancy":{
                    "score": "<score>",
                    "reason": "<reason>"
                    }
                }
            },
        ...
    ]
    """

    with open("test_data_score.json") as f:
        data = json.load(f)
    with open("test_data_score_searchflight.json") as f:
        data_searchflight = json.load(f)
    data.extend(data_searchflight)

    # load model output
    with open("../outputs/add_thought_model_output/output.json") as f:
        model_output = json.load(f)

    # load baseline output
    with open("../outputs/salesbot1_baseline/output.json") as f:
        baseline_output = json.load(f)

    with open("../outputs/add_thought_model_output_searchflight/output.json") as f:
        model_output_searchflight = json.load(f)

    # load baseline output
    with open("../outputs/salesbot1_baseline_searchflight/output.json") as f:
        baseline_output_search_flight = json.load(f)

    model_output.extend(model_output_searchflight)
    baseline_output.extend(baseline_output_search_flight)
    
    #get histogram of the score of naturalness and the consistancy as a dictionary the interval is 10
    naturalness_score = []
    consistancy_score = []
    for i in range(len(data)):
        naturalness_score.append(data[i]["score"][0]["naturalness"]["score"])
        consistancy_score.append(data[i]["score"][0]["consistancy"]["score"])
    naturalness_score = [int(i) for i in naturalness_score]
    consistancy_score = [int(i) for i in consistancy_score]
    naturalness_score_hist = {}
    consistancy_score_hist = {}
    for i in range(11):
        naturalness_score_hist[i] = 0
        consistancy_score_hist[i] = 0
    for i in naturalness_score:
        # add 1 in every 1-10
        naturalness_score_hist[i//10] += 1
    for i in consistancy_score:
        # add 1 in every 1-10
        consistancy_score_hist[i//10] += 1
    print("naturalness_score_hist: ", naturalness_score_hist)
    print("consistancy_score_hist: ", consistancy_score_hist)

    type_thought = ["chitchat", "transition", "continuation", "proceed to tod"]
    intents = ['None','FindAttraction', 'FindRestaurants', 'FindMovie', 'LookupMusic', 'SearchHotel', 'FindEvents', 'GetTransportation', 'SearchFlights']
    id_to_score = {}
    model_intent_match_cnt_each = {i:{j:0 for j in intents} for i in intents}
    model_type_match_cnt_each = {i:{j:0 for j in type_thought} for i in type_thought}
    model_exact_match_cnt = 0
    for d in data:
        id = "_".join(d["id"].split("_")[:2])
        id_to_score[id] = d["score"][0]

    for t in model_output:
        id = "_".join(t["id"].split("_")[:2])
        score = id_to_score[id]
        t["score"] = score
        if int(score["naturalness"]["score"]) < 90 or int(score["consistancy"]["score"]) < 90:
            continue
        # resp
        resp = t["response"]["value"]
        resp_thought = resp.split("Response")[0].split("Thought: ")[-1].strip()
        resp_intent = ""
        resp_type = ""
        for intent in intents:
            if intent in resp_thought:
                resp_intent = intent
                break
        if resp_intent == "":
            resp_intent = "None"
            resp_type = "chitchat"
        elif "continue" in resp_thought:
            resp_type = "continuation"
        elif  "Proceed to task oriented dialog agent" in resp:
            resp_type = "proceed to tod"
        else:
            resp_type = "transition"

        gt_resp = t["conversations"][1]["value"]
        gt_resp_thought = gt_resp.split("Response")[0].split("Thought: ")[-1].strip()
        gt_resp_intent = ""
        gt_resp_type = ""
        for intent in intents:
            if intent in gt_resp_thought:
                gt_resp_intent = intent
                break
        if gt_resp_intent == "":
            gt_resp_intent = "None"
            gt_resp_type = "chitchat"
        elif "continue" in gt_resp_thought:
            gt_resp_type = "continuation"
        elif  "Proceed to task oriented dialog agent" in gt_resp:
            gt_resp_type = "proceed to tod"
        else:
            gt_resp_type = "transition"

        model_intent_match_cnt_each[gt_resp_intent][resp_intent] += 1
        model_type_match_cnt_each[gt_resp_type][resp_type] += 1
        if gt_resp_intent == resp_intent and gt_resp_type == resp_type:
            model_exact_match_cnt += 1

    # baseline
    baseline_intent_match_cnt_each = {i:{j:0 for j in intents} for i in intents}
    baseline_type_match_cnt_each = {i:{j:0 for j in type_thought} for i in type_thought}
    baseline_exact_match_cnt = 0
    for t in baseline_output:
        id = "_".join(t["id"].split("_")[:2])
        score = id_to_score[id]
        if int(score["naturalness"]["score"]) < 90 or int(score["consistancy"]["score"]) < 90:
            continue
        # resp
        resp = t["response"]["value"]
        resp_thought = resp.split("Response")[0].split("Thought: ")[-1].strip()
        resp_intent = ""
        resp_type = ""
        for intent in intents:
            if intent in resp_thought:
                resp_intent = intent
                break
        if resp_intent == "": 
            resp_intent = "None"
            resp_type = "chitchat"
        elif "continue" in resp_thought:
            resp_type = "continuation"
        elif  "Proceed to task oriented dialog agent" in resp:
            resp_type = "proceed to tod"
        else:
            resp_type = "transition"

        gt_resp = t["conversations"][1]["value"]
        gt_resp_thought = gt_resp.split("Response")[0].split("Thought: ")[-1].strip()
        gt_resp_intent = ""
        gt_resp_type = ""
        for intent in intents:
            if intent in gt_resp_thought:
                gt_resp_intent = intent
                break
        if gt_resp_intent == "":
            gt_resp_intent = "None"
            gt_resp_type = "chitchat"
        elif "continue" in gt_resp_thought:
            gt_resp_type = "continuation"
        elif  "Proceed to task oriented dialog agent" in gt_resp:
            gt_resp_type = "proceed to tod"
        else:
            gt_resp_type = "transition"

        baseline_intent_match_cnt_each[gt_resp_intent][resp_intent] += 1
        baseline_type_match_cnt_each[gt_resp_type][resp_type] += 1
        if gt_resp_intent == resp_intent and gt_resp_type == resp_type:
            baseline_exact_match_cnt += 1

    # print result prettily 
    print("Model Intent Match Count Each: ")
    model_total_intent_match_cnt = 0
    model_total_type_match_cnt = 0
    for k,v in model_intent_match_cnt_each.items():
        print("     Ground Truth Intent: ", k)
        sum = 0
        for v_k, v_v in v.items():
            print(f"        Output Intent: {v_k}, Count: {v_v}")
            sum += v_v
        model_total_intent_match_cnt += v[k]
        print("Match Rate", v[k]/sum)
    for k,v in model_type_match_cnt_each.items():
        print("     Ground Truth Type: ", k)
        sum = 0
        for v_k, v_v in v.items():
            print(f"        Output Type: {v_k}, Count: {v_v}")
            sum += v_v
        model_total_type_match_cnt += v[k]
        print("Match Rate", v[k]/sum)


    print("Model Total Intent Match Count: ", model_total_intent_match_cnt)
    print("Model Total Type Match Count: ", model_total_type_match_cnt)
    print("Model Total Intent Match Rate: ", model_total_intent_match_cnt/len(model_output))
    print("Model Total Type Match Rate: ", model_total_type_match_cnt/len(model_output))
    print("Model Exact Match Count: ", model_exact_match_cnt)
    print("Model Exact Match Rate: ", model_exact_match_cnt/len(model_output))


    print("Baseline Intent Match Count Each: ")
    baseline_total_intent_match_cnt = 0
    baseline_total_type_match_cnt = 0
    for k,v in baseline_intent_match_cnt_each.items():
        if k == "LookUpMusic":
            continue
        print("     Ground Truth Intent: ", k)
        sum = 0
        for v_k, v_v in v.items():
            print(f"        Output Intent: {v_k}, Count: {v_v}")
            sum += v_v
        print("Match Rate", v[k]/sum)
        baseline_total_intent_match_cnt += v[k]
    for k,v in baseline_type_match_cnt_each.items():
        print("     Ground Truth Type: ", k)
        sum = 0
        for v_k, v_v in v.items():
            print(f"        Output Type: {v_k}, Count: {v_v}")
            sum += v_v
        print("Match Rate", v[k]/sum)
        baseline_total_type_match_cnt += v[k]

    print("Baseline Total Intent Match Count: ", baseline_total_intent_match_cnt)
    print("Baseline Total Type Match Count: ", baseline_total_type_match_cnt)
    print("Baseline Total Intent Match Rate: ", baseline_total_intent_match_cnt/len(baseline_output))
    print("Baseline Total Type Match Rate: ", baseline_total_type_match_cnt/len(baseline_output))
    print("Baseline Exact Match Count: ", baseline_exact_match_cnt)
    print("Baseline Exact Match Rate: ", baseline_exact_match_cnt/len(baseline_output))

    # count number of data with different id and score is higher than 90
    cnt = 0
    for d in data:
        if int(d["score"][0]["naturalness"]["score"]) >= 90 and int(d["score"][0]["consistancy"]["score"]) >= 90:
            # if intent is not GetTransportation, SearchFlights
            thought = d["conversations"][1]["value"].split("Response")[0].split("Thought: ")[-1].strip()
            if "GetTransportation" not in thought and "SearchFlights" not in thought:
                cnt += 1
    print("Number of data with score higher than 90: ", cnt)
    print("Number of data: ", len(data))










    




    

if __name__ == "__main__":
    main()

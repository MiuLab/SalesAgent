from utils import read_json_file, write_json_file

data = read_json_file('./MSGD_dataset_final.json')

#check if utterance in chitchat_context is overlapped in dialog, if not print the difference and id
bad_id_set = {}
for d in data:
    if d['chitchat_context'] != []:
        for i, c in enumerate(d['chitchat_context']):
            if c not in d['dialog']:
                print(d['id'])
                print(f"chitchat_context: {c}")
                print(f"dialog: {d['dialog']}")
                if d['id'] not in bad_id_set:
                    bad_id_set[d['id']] = [i]
                else:
                    bad_id_set[d['id']].append(i)
            elif d['id'] in bad_id_set and i != d['dialog'].index(c) + len(bad_id_set[d['id']]):
                print(d['id'])
                print(f"chitchat_context: {c}")
                print(f"dialog: {d['dialog']}")
                if d['id'] not in bad_id_set:
                    bad_id_set[d['id']] = [i]
                else:
                    bad_id_set[d['id']].append(i)


for k, v in bad_id_set.items():
    print (f"{k}: {v}")

# fix the problem
cnt = 0
for d in data:
    if d['id'] in bad_id_set:
        idx_ls = [idx for idx in bad_id_set[d['id']]]
        if 0 not in idx_ls:
            d['chitchat_context'] = d['dialog'][:len(d['chitchat_context'])]
            continue
        missing_context = []
        for i in range(len(idx_ls)):
            if i != 0 and idx_ls[i] - idx_ls[i-1] != 1:
                break
            missing_context.append(d['chitchat_context'][idx_ls[i]])

        d['dialog'] = missing_context + d['dialog']
        d['chitchat_context'] = d['dialog'][:len(d['chitchat_context'])]
        cnt += 1
print(cnt)
from pprint import pprint
for d in data:
#check if utterance are said by the same speaker:
    if d['dialog'] != []:
        for i in range(len(d['dialog'])-1):
            if "User:" in d['dialog'][i] and "User:" in d['dialog'][i+1]:
                pprint(d['id'])
                pprint(f"dialog: {d['dialog']}")
            elif "Agent:" in d['dialog'][i] and "Agent:" in d['dialog'][i+1]:
                pprint(d['id'])
                pprint(f"dialog: {d['dialog']}")
# check if all dialog start with User:
# write_json_file('./MSGD_dataset_final_sanity_1024.json', data)
# check transition_sentence is in dialog and not in first positions
# bad_id_set_not_in = set()
# bad_id_set_first = set()
# for d in data:
#     if d['transition_sentence']['utterance'] not in d['dialog']:
#         bad_id_set_not_in.add(d['id'])
#     if d['transition_sentence']['utterance'] == d['dialog'][0]:
#         bad_id_set_first.add(d['id'])
#
# print(f"first:{bad_id_set_first}")
# print(len(bad_id_set_first))
# intent_set = set()
# desc_set = set()
# for d in data:
#     intent_set.add(d['intent']['type'])
#     desc_set.add(d['intent']['description'])
# print(intent_set)
# print(desc_set)
# print(len(data))
#
# # check if transition sentences are more than 1 in dialog
# for d in data:
#     if d['dialog'].count(d['transition_sentence']['utterance']) > 1:
#         print(d['id'])

# check if all chitchat_context are end with user
# for d in data:
#     if not d['transition_sentence']['utterance'].startswith("User:"):
#         print(d["id"])
#

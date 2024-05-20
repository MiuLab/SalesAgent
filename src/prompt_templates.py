MSGD_PATH = "salesbot2_prompt/Merge_SGD.json"
STOD_PATH = "salesbot2_prompt/TOD_Simulators.json"
CONTINUE_PREFIX = (
    "Here is the potential intent(with description) and an incomplete dialogue:\n"
)

CONTINUE_SUFFIX = "Your goal as following: \n \
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

REWRITE_PREFIX = "You will be given a conversation between two people.\n \
\n \
 \n \
Here is what you should do:\n \
1. Identify the inconsistent utterances.\n \
2. Give some reasons why they are inconsistent.\n \
3. Modify the dialogue based on previous identified utterances\n \
4. The rewritten dialogue should be more than 6 turns\n \
\n \
Here is the conversation: \n"


REWRITE_SUFFIX = "You MUST follow the format as :\n \
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

INTENT_DET_PREFIX = (
    "You will be given a dialogue and a list of topics of conversation.\n \
Please tell me which of the following topics will be the most reasonable one to be pivoted to in the dialogue.\n \
\n \
\n \
Here is the dialogue:\n"
)

INTENT_DET_SUFFIX = "Here is the list of topics:\n \
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

TRANS_DET_PREFIX = "You will be given a dialogue below and a potential intent below,"
TRANS_DET_SUFFIX = " and your goal is a following:\n \
1. Identify the first utterance that apparently mentions the intent given above\n \
2. You should choose only one turn in the given dialogue\n \
3. The chosen turn should be said by User\n \
\n \
Please follow the output format as below:\n \
\n \
The chosen turn:\n \
..."

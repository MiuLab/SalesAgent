# SalesAgent

## Generate SalesBot 2.0 Datasets from Scratch
### 1. Install the required packages
```bash
pip install -r requirements.txt
```
### 2. Rewrite
```bash
python generate_prompts.py --mode "generate_rewrite" --input_file "./salesbot2_prompt/Merge_SGD.json" --prompt_dir "<path_to_save_prompts>"
python generate_prompts.py --mode "rewrite" --input_file "<output_file_from_generate_rewrite>" --output_dir "<dir_to_save_outputs_from_llm>"
```
### 3. Intent Detection
```bash
python generate_prompts.py --mode "generate_intent" --prev_output_dir "<output_dir_from_llm>" --prompt_dir "<path_to_save_prompts>"
python generate_prompts.py --mode "intent_detection" --input_file "<output_file_from_generate_intent>" --output_dir "<path_to_save_outputs_from_llm>"
```
### 4. Dialogue Continuation
```bash
python generate_prompts.py --mode "generate_continue" --prev_output_dir "<output_dir_from_llm>" --prompt_dir "<path_to_save_prompts>"
python generate_prompts.py --mode "dialogue_continuation" --input_file "<output_file_from_generate_dialogue>" --output_dir "<path_to_save_outputs_from_llm>"
```
### 5. Transtion Detection
```bash
python generate_prompts.py --mode "generate_trans" --prev_output_dir "<output_dir_from_llm>" --prompt_dir "<path_to_save_prompts>"
python generate_prompts.py --mode "transition_detection" --input_file "<output_file_from_generate_transition>" --output_dir "<path_to_save_outputs_from_llm>"
```
### 6. Generate SalesBot 2.0 Dataset
```bash
python generate_prompts.py --mode "generate_dataset" --prev_output_dir "<output_dir_from_llm>" --prompt_dir "<path_to_save_prompts>"
```
## Prepare Training and Evaluation Data
```bash
cd ./create_dataset/utils/
python create_salesbot2_dataset.py
python create_salesbot1_dataset.py
```
## Fine-tune Llama-2 on Dataset
```bash
cd scripts/
bash run_finetune.sh <path_to_dataset> <path_to_save_model>
```



## TODO:
- [x] Push Raw materials
- [x] Remove all API keys
- [ ] Remove useless files
- [ ] Organize the project and code
- [ ] Add a proper license
- [ ] Add a proper README
- [ ] Referece the original project
- [x] Add model source

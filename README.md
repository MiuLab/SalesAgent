# Injecting Salesperson’s Dialogue Strategies in Large Language Models with Chain-of-Thought Reasoning
This repo is the implementation of the paper [Injecting Salesperson’s Dialogue Strategies in Large Language Models with Chain-of-Thought Reasoning](https://arxiv.org/pdf/2404.18564), including the dataset SalesBot 2.0 (revised from [SalesBot 1.0](https://github.com/MiuLab/SalesBot)) and the fine-tuned model.

Please cite the following reference if you use the released data or model.

```
@inproceedings{chang2024injecting,
      title={Injecting Salesperson's Dialogue Strategies in Large Language Models with Chain-of-Thought Reasoning}, 
      author={Wen-Yu Chang and Yun-Nung Chen},
      year={2024},
      booktitle={Findings of the Association for Computational Linguistics: ACL 2024}
}
```


## Model
You can find the model (llama-2-7B) fine-tuned on SalesBot 2.0 data [here](https://huggingface.co/miulab/SalesBot2_CoT).
### Interaction with the model
```bash
cd scripts/
python interactive.py --model_path <path_to_model> --temperature 0.7
```
## Generate SalesBot 2.0 Datasets and Train Model from Scratch
![image](https://github.com/MiuLab/SalesAgent/assets/2268109/1569238d-aa01-497f-9d2a-f5cec101f6ad)

### 1. Install the required packages
```bash
pip3 install openai, tqdm, fast
cd ./FastChat/
pip3 install -e ".[model_worker,webui,train]"
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



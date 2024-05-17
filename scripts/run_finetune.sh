data_path = $1
output_dir = $2
torchrun --nproc_per_node=4 --master_port=20001 ./FastChat/fastchat/train/train_xformers.py \
  --model_name_or_path "meta-llama/Llama-2-7b-chat-hf"  \
  --data_path $(data_path) \
  --bf16 False \
  --fp16 True \
  --num_train_epochs 10 \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size 1 \
  --gradient_accumulation_steps 32 \
  --evaluation_strategy no \
  --save_strategy epoch \
  --save_total_limit 1 \
  --learning_rate 2e-5 \
  --weight_decay 0.0 \
  --warmup_ratio 0.04 \
  --lr_scheduler_type cosine \
  --logging_steps 5 \
  --fsdp 'full_shard auto_wrap' \
  --fsdp_transformer_layer_cls_to_wrap LlamaDecoderLayer \
  --tf32 False \
  --model_max_length 1024 \
  --gradient_checkpointing True \
  --output_dir $(output_dir)\


python -m vllm.entrypoints.api_server \
        --host="140.112.29.239" \
        --port=5000 \
        --model="meta-llama/Llama-2-7b-chat-hf" \
        --dtype="float16" \
        --tensor-parallel-size=2 \

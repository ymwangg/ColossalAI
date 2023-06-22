export WANDB_MODE=offline
colossalai run --nproc_per_node 8 train_llama.py \
    --model_name_or_path ~/models/llama-7b-with-tokenizer/ \
    --data_path ./alpaca_data.json \
    --output_dir ./trained/saved_model.pt \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 1 \
    --learning_rate 2e-6 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \

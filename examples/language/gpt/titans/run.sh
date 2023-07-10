export DATA=./gpt_data/small-gpt-dataset.json
DUMMY_DATA=--use_dummy_dataset
colossalai run --nproc_per_node=8 train_gpt.py --config ./configs/gpt2_small_zero3_pp1d.py --from_torch
# colossalai run --nproc_per_node=8 train_gpt.py --config ./configs/gpt3_zero3_pp1d.py --from_torch
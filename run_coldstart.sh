python main.py \
  --train_json_file "./data/train_cold_start.json" \
  --meta_json_file "./data/meta_MMMU_Pro.json" \
  --test_json_file "./data/test_MMMU.json" \
  --reward_model "OpenGVLab/InternVL3-1B" \
  --weights_path "./weights_cold_start" \
  --lr 5e-5 \
  --weight_decay 5e-2 \
  --unroll_steps 1 \
  --initialization 1.0 \
  --save_every_iterations 5000 \
  --retrain 1 \
  --iteration_num 20000


export PATH=/home/vnmember05/envs/cuda-12.8/bin:$PATH
export LD_LIBRARY_PATH=/home/vnmember05/envs/cuda-12.8/lib64:$LD_LIBRARY_PATH
export CUDA_HOME="/home/vnmember05/envs/cuda-12.8"

export CC=gcc
export CXX=g++

python clean_video.py \
  --data-root /project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/InternData-A1 \
  --output-json /project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/InternData-A1/clean_results/final_frame_errors_test.jsonl \
  --subset /project/vonneumann1/wcy/code/starVLA-dev/playground/Datasets/InternData-A1/sim_updated/pick_and_place_tasks/split_aloha/parallel_pick_and_place_right_right/parallel_pick_and_place_right_right/dish \
  --model-path /project/vonneumann1/wcy/models/LLM/Qwen3.5-122B-A10B \
  --tp-size 8 \
  --context-length 32768 \
  --max-workers 4 \
  --limit 200


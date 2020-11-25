python3 -m ermas.eval --random_seed 0 --exp_id da --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 1 --exp_id da --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 2 --exp_id da --main_episodes 1 --num_loops 1000 --perturb 0.3 &
wait

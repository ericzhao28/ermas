python3 -m ermas.eval --random_seed 0 --exp_id ermas5 --main_episodes 1 --num_loops 1000 --perturb 0.1 &
python3 -m ermas.eval --random_seed 1 --exp_id ermas5 --main_episodes 1 --num_loops 1000 --perturb 0.1 &
python3 -m ermas.eval --random_seed 2 --exp_id ermas5 --main_episodes 1 --num_loops 1000 --perturb 0.1 &
python3 -m ermas.eval --random_seed 0 --exp_id rr --main_episodes 1 --num_loops 1000 --perturb 0.1 &
python3 -m ermas.eval --random_seed 1 --exp_id rr --main_episodes 1 --num_loops 1000 --perturb 0.1 &
python3 -m ermas.eval --random_seed 2 --exp_id rr --main_episodes 1 --num_loops 1000 --perturb 0.1 &
wait

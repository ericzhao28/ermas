python3 -m ermas.eval --random_seed 0 --exp_id wa05 --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 1 --exp_id wa05 --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 2 --exp_id wa05 --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 0 --exp_id wa25 --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 1 --exp_id wa25 --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 0 --exp_id wa25 --main_episodes 1 --num_loops 1000 --perturb 0.3 &
wait
exit
python3 -m ermas.eval --random_seed 1 --exp_id rr --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 2 --exp_id rr --main_episodes 1 --num_loops 1000 --perturb 0.3 &
python3 -m ermas.eval --random_seed 2 --exp_id rr --main_episodes 1 --num_loops 1000 --perturb 0.3 &
wait

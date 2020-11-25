python3 -m ermas.train --random_seed 0 --exp_id da --main_episodes 1 --num_loops 10000 --da_random 4 &
python3 -m ermas.train --random_seed 1 --exp_id da --main_episodes 1 --num_loops 10000 --da_random 4 &
python3 -m ermas.train --random_seed 2 --exp_id da --main_episodes 1 --num_loops 10000 --da_random 4 &
wait

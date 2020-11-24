python3 -m ermas.train --random_seed 0 --exp_id wa05 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.05 &
python3 -m ermas.train --random_seed 1 --exp_id wa05 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.05 & 
python3 -m ermas.train --random_seed 2 --exp_id wa05 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.05 &
python3 -m ermas.train --random_seed 0 --exp_id wa25 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.25 &
python3 -m ermas.train --random_seed 1 --exp_id wa25 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.25 & 
python3 -m ermas.train --random_seed 2 --exp_id wa25 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.25 &
wait

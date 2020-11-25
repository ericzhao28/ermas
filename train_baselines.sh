python3 -m ermas.train --random_seed 0 --exp_id rr --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --crra_sigma 0.4 &
python3 -m ermas.train --random_seed 1 --exp_id rr --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --crra_sigma 0.4 & 
python3 -m ermas.train --random_seed 2 --exp_id rr --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --crra_sigma 0.4 &
python3 -m ermas.train --random_seed 0 --exp_id wa05 --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.05 &
python3 -m ermas.train --random_seed 1 --exp_id wa05 --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.05 & 
python3 -m ermas.train --random_seed 2 --exp_id wa05 --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.05 &
python3 -m ermas.train --random_seed 0 --exp_id wa25 --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.25 &
python3 -m ermas.train --random_seed 1 --exp_id wa25 --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.25 & 
python3 -m ermas.train --random_seed 2 --exp_id wa25 --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --worst_action_prob 0.25 &
python3 -m ermas.train --random_seed 0 --exp_id da --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --da_random 4 &
python3 -m ermas.train --random_seed 1 --exp_id da --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --da_random 4 &
python3 -m ermas.train --random_seed 2 --exp_id da --resume_exp_id marl6 --main_episodes 1 --num_loops 10000 --da_random 4 &
wait

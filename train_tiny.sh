python3 -m ermas.train --random_seed 0 --exp_id marl6 --main_episodes 1 --num_loops 10000 &
python3 -m ermas.train --random_seed 1 --exp_id marl6 --main_episodes 1 --num_loops 10000 &
python3 -m ermas.train --random_seed 2 --exp_id marl6 --main_episodes 1 --num_loops 10000 &
wait
exit
python3 -m ermas.train --random_seed 0 --exp_id marl5 --main_episodes 1 --num_loops 10000 &
python3 -m ermas.train --random_seed 1 --exp_id marl5 --main_episodes 1 --num_loops 10000 &
python3 -m ermas.train --random_seed 2 --exp_id marl5 --main_episodes 1 --num_loops 10000 &
wait
exit
python3 -m ermas.train --random_seed 0 --exp_id marl4 --main_episodes 1 --num_loops 10000 &
python3 -m ermas.train --random_seed 1 --exp_id marl4 --main_episodes 1 --num_loops 10000 &
python3 -m ermas.train --random_seed 2 --exp_id marl4 --main_episodes 1 --num_loops 10000 &
wait

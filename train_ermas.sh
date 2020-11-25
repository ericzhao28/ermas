python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas8 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 3000.0 --beta 0.2 &
python3 -m ermas.ermas_train --random_seed 1 --exp_id ermas8 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 3000.0 --beta 0.2 &
python3 -m ermas.ermas_train --random_seed 2 --exp_id ermas8 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 3000.0 --beta 0.2 &
wait
exit
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas7 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 400.0 --beta 0.2 &
python3 -m ermas.ermas_train --random_seed 1 --exp_id ermas7 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 400.0 --beta 0.2 &
python3 -m ermas.ermas_train --random_seed 2 --exp_id ermas7 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 400.0 --beta 0.2 &
wait
exit
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas6 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 1000.0 --beta 0.2 &
python3 -m ermas.ermas_train --random_seed 1 --exp_id ermas6 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 1000.0 --beta 0.2 &
python3 -m ermas.ermas_train --random_seed 2 --exp_id ermas6 --lambda_lr 0.01 --initial_lambda 1 --ermas_eps 1000.0 --beta 0.2 &
wait

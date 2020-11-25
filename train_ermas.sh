python3 -m ermas.ermas_train --random_seed 1 --exp_id ermas33 --lambda_lr 10 --ermas_eps 0.4 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 1 --exp_id ermas35 --lambda_lr 10 --ermas_eps -0.5 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 2 --exp_id ermas33 --lambda_lr 10 --ermas_eps 0.4 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 2 --exp_id ermas35 --lambda_lr 10 --ermas_eps -0.5 --resume_exp_id marl6 &
wait
exit
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas31 --lambda_lr 10 --ermas_eps 1.0 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas32 --lambda_lr 10 --ermas_eps 0.2 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas33 --lambda_lr 10 --ermas_eps 0.4 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas34 --lambda_lr 10 --ermas_eps -1.0 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas35 --lambda_lr 10 --ermas_eps -0.5 --resume_exp_id marl6 &
wait
exit
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas21 --lambda_lr 1 --ermas_eps 1.0 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas22 --lambda_lr 1 --ermas_eps 0.2 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas23 --lambda_lr 1 --ermas_eps 0.4 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas24 --lambda_lr 1 --ermas_eps -1.0 --resume_exp_id marl6 &
python3 -m ermas.ermas_train --random_seed 0 --exp_id ermas25 --lambda_lr 1 --ermas_eps -0.5 --resume_exp_id marl6 &
wait

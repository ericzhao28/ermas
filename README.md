# ERMAS Implementation
This repository attempts to re-implement the ERMAS algorithm described in [this paper](https://openreview.net/forum?id=uIc4W6MtbDA). 

## Consumer-firm problem setting
This environment simulates the seasonal behaviors of consumers and firms. The complexity of this problem lies between the bimatrix game and the complex spatiotemporal tax simulation. 

We simulate three types of agents: (1) consumers, (2) suppliers/firms, and (3) shipping companies. Consumer demand is seasonal (sinusoidal). Suppliers can produce up to X products a day and set market prices. Both suppliers and consumers can hold inventory, but supplier inventory incurs a linear cost while consumer inventory is free but limited. Each day, shipping companies may charge a shipping fee proportional to the square root of the amount of shipped product. Here, we seek to learn a supplier/firm policy (the planner) for setting product costs and production rates. In particular, the policy should be robust to uncertainty about the consumer utility functions.

## Instructions
First, fill in a config file under `./ermas/config.py` with an API key from [Comet.ml](https://comet.ml):
```
comet_ml_key = "aovevijawievjij"
```
To train MARL, run `train_marl.sh`. After completion, run `train_ermas.sh` and `train_baselines.sh`.
To evaluate, run `eval_ermas.sh` and `eval_baselines.sh`.

## License
See the included `License` file.

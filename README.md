# LLM-GCBF+ for deadlock resolution

Jax Official Implementation of Paper: [K Garg*](https://kunalgarg.mit.edu/), [J Arkin](https://aeroastro.mit.edu/realm/team/jake-arkin/), [S Zhang](https://syzhang092218-source.github.io), [N Roy](https://groups.csail.mit.edu/rrg/index.php?n=Main.HomePage), [C Fan](https://chuchu.mit.edu): "[Large Language Models to the Rescue: Deadlock Resolution in Multi-Robot Systems](https://mit-realm.github.io/LLM-gcbfplus-website/)". 

## Dependencies

We recommend to use [CONDA](https://www.anaconda.com/) to install the requirements:

```bash
conda create -n gcbfplus python=3.10
conda activate gcbfplus
cd gcbfplus
```

Then install jax following the [official instructions](https://github.com/google/jax#installation), and then install the rest of the dependencies:
```bash
pip install -r requirements.txt
```

## Installation

Install GCBF: 

```bash
pip install -e .
```

## Run

### High-level planner for deadlock resolution
To run the high-level planner for deadlock resolution, use:

```bash
python -u  test_with_LLM.py --path logs/SingleIntegrator/gcbf+/model_with_traj/seed0_20240227110346 -n $N --epi 20 --obs $obs --max-step $ms --area-size 4  --keep-mode $km --nojit-rollout --num-incontext-prompts $k --leader_model $gpt --num_LLM_calls $num_calls

```

where the flags are:
- `-n`: number of agents
- `--obs`: number of obstacles
- `--area-size`: side length of the environment
- `--max-step`: maximum number of steps for each episode, increase this if you have a large environment
- `--path`: path to the log folder
- `--keep-mode`: keep mode for the high-level planner
- `--num-incontext-prompts`: number of in-context examples
- `--leader_model`: leader model for the high-level planner including 
    - 'gpt3.5'
    - 'gpt4'
    - 'hand' for hand-designed heuristic leader-assignment
    - 'fixed' for fixed leader-assignment
    - 'random' for random leader-assignment
    - 'none' for no leader-assignment
- `--num_LLM_calls`: number of LLM calls for "Ensemble" implementation of the high-level planner

For testing on "Randomized room" environment, use:

```bash
python -u  test_with_LLM.py --path logs/SingleIntegrator/gcbf+/model_with_traj/seed0_20240227110346/ -n 1 --epi 20 --obs 1 --preset_reset --preset_scene 'rand box' --max-step $ms --area-size 1 --keep-mode $km --nojit-rollout --num-incontext-prompts $k --leader_model $gpt --num_LLM_calls $num_calls 
```
where 
-`--preset_reset` is used to reset the environment to a fixed initial state from
    - 'rand box' for a random room environment
    - 'original box' for a fixed room environment
    - 'box' for room-like environment with more obstacles.

### GCBF+ low-level controller for safe multi-agent navigation

### Environments

We provide 3 2D environments including `SingleIntegrator`, `DoubleIntegrator`, and `DubinsCar`, and 2 3D environments including `LinearDrone` and `CrazyFlie`.

### Algorithms

We provide algorithms including GCBF+ (`gcbf+`), GCBF (`gcbf`), centralized CBF-QP (`centralized_cbf`), and decentralized CBF-QP (`dec_share_cbf`). Use `--algo` to specify the algorithm. 

### Hyper-parameters

To reproduce the results shown in our paper, one can refer to [`settings.yaml`](./settings.yaml).

### Train

To train the model (only GCBF+ and GCBF need training), use:

```bash
python train.py --algo gcbf+ --env DoubleIntegrator -n 8 --area-size 4 --loss-action-coef 1e-4 --n-env-train 16 --lr-actor: 1e-5 --lr-cbf: 1e-5 --horizon: 32
```

In our paper, we use 8 agents with 1000 training steps. The training logs will be saved in folder `./logs/<env>/<algo>/seed<seed>_<training-start-time>`. We also provide the following flags:

- `-n`: number of agents
- `--env`: environment, including `SingleIntegrator`, `DoubleIntegrator`, `DubinsCar`, `LinearDrone`, and `CrazyFlie`
- `--algo`: algorithm, including `gcbf`, `gcbf+`
- `--seed`: random seed
- `--steps`: number of training steps
- `--name`: name of the experiment
- `--debug`: debug mode: no recording, no saving
- `--obs`: number of obstacles
- `--n-rays`: number of LiDAR rays
- `--area-size`: side length of the environment
- `--n-env-train`: number of environments for training
- `--n-env-test`: number of environments for testing
- `--log-dir`: path to save the training logs
- `--eval-interval`: interval of evaluation
- `--eval-epi`: number of episodes for evaluation
- `--save-interval`: interval of saving the model

In addition, use the following flags to specify the hyper-parameters:
- `--alpha`: GCBF alpha
- `--horizon`: GCBF+ look forward horizon
- `--lr-actor`: learning rate of the actor
- `--lr-cbf`: learning rate of the CBF
- `--loss-action-coef`: coefficient of the action loss
- `--loss-h-dot-coef`: coefficient of the h_dot loss
- `--loss-safe-coef`: coefficient of the safe loss
- `--loss-unsafe-coef`: coefficient of the unsafe loss
- `--buffer-size`: size of the replay buffer

### Test

To test the learned model, use:

```bash
python test.py --path <path-to-log> --epi 5 --area-size 4 -n 16 --obs 0
```

This should report the safety rate, goal reaching rate, and success rate of the learned model, and generate videos of the learned model in `<path-to-log>/videos`. Use the following flags to customize the test:

- `-n`: number of agents
- `--obs`: number of obstacles
- `--area-size`: side length of the environment
- `--max-step`: maximum number of steps for each episode, increase this if you have a large environment
- `--path`: path to the log folder
- `--n-rays`: number of LiDAR rays
- `--alpha`: CBF alpha, used in centralized CBF-QP and decentralized CBF-QP
- `--max-travel`: maximum travel distance of agents
- `--cbf`: plot the CBF contour of this agent, only support 2D environments
- `--seed`: random seed
- `--debug`: debug mode
- `--cpu`: use CPU
- `--u-ref`: test the nominal controller
- `--env`: test environment (not needed if the log folder is specified)
- `--algo`: test algorithm (not needed if the log folder is specified)
- `--step`: test step (not needed if testing the last saved model)
- `--epi`: number of episodes to test
- `--offset`: offset of the random seeds
- `--no-video`: do not generate videos
- `--log`: log the results to a file
- `--dpi`: dpi of the video
- `--nojit-rollout`: do not use jit to speed up the rollout, used for large-scale tests

To test the nominal controller, use:

```bash
python test.py --env SingleIntegrator -n 16 --u-ref --epi 1 --area-size 4 --obs 0
```

To test the CBF-QPs, use:

```bash
python test.py --env SingleIntegrator -n 16 --algo dec_share_cbf --epi 1 --area-size 4 --obs 0 --alpha 1
```

### Pre-trained models

We provide the pre-trained models in the folder [`logs`](logs).

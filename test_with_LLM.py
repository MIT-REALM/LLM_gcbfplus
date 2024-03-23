import argparse
import datetime
import functools as ft
import os
import pathlib
import ipdb
import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import yaml
import pickle

from gcbfplus.algo import GCBF, GCBFPlus, make_algo, CentralizedCBF, DecShareCBF
from gcbfplus.env import make_env
from gcbfplus.env.base import RolloutResult
from gcbfplus.env.utils import get_leader_id_dir, TrajLog
from gcbfplus.trainer.utils import get_bb_cbf
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.utils.utils import jax_jit_np, tree_index, jax_vmap
# from gcbfplus.utils.typing import Array

from gcbfplus.env.utils import TrajLog

from gcbfplus.utils.LLM_utils import create_init_prompt

def test(args):
    if args.leader_model =='none':
        args.use_leader = False
        args.use_llm_leader = False
    else:
        args.use_leader = True
        args.use_llm_leader = True

    if args.preset_reset and (args.preset_scene == 'box' or args.preset_scene == 'rand box' or args.preset_scene == 'original box'):
        stat_file = 'LLM_files/LLM_box_stats.csv'
        if not os.path.exists(stat_file):
            with open(stat_file, "w") as f:
                f.write('GPT model, Mean init distance, num_incontext, num_runtime_incontext, num_agents, num_obstacles, finish_count, Num LLM calls, Distance traveled, Time taken, Tokens sent, Tokens received, Mean final distance \n')
    else:
        stat_file = 'LLM_files/LLM_random_stats.csv'
        if not os.path.exists(stat_file):
            with open(stat_file, "w") as f:
                f.write('GPT model, Mean init distance, num_incontext, num_runtime_incontext, num_agents, num_obstacles, finish_count, Num LLM calls, Distance traveled, Time taken, Tokens sent, Tokens received, Mean final distance \n')
     
    # os.makedirs(stat_file, exist_ok=True)
    
    print(f"> Running test.py {args}")
    # set up environment variables and seed
    os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
    if args.cpu:
        os.environ["JAX_PLATFORM_NAME"] = "cpu"
    if args.debug:
        jax.config.update("jax_disable_jit", True)

    stamp_str = datetime.datetime.now().strftime("%m%d-%H%M")
    num_incontext = args.num_incontext_prompts
    
    prompts, log_dir = create_init_prompt(num_incontext, args.preset_reset)
    log_dir = os.path.join(log_dir, str(args.preset_scene))
    if args.leader_model == 'none':
        log_dir = os.path.join(log_dir, 'GCBF+')
    else:
        log_dir = os.path.join(log_dir, args.leader_model)
    log_dir = os.path.join(log_dir, stamp_str)
    os.makedirs(log_dir, exist_ok=True)
    with open(f"{log_dir}/log_message.txt", "a") as f:
        f.write('Model path: ')
        f.write(args.path)
        f.write('\n')
        f.write('Test settings: ')
        f.write(str(args))
        f.write('\n')
    if args.num_incontext_prompts > 0:
        np.random.seed(args.seed + args.particular_epi)

        # if not args.preset_reset:
        #     incontext_prompt_file = os.path.join(args.path, 'LLM_incontext_files/LLM_all_GPT_prompts.txt')
        #     incontext_output_file = os.path.join(args.path, 'LLM_incontext_files/LLM_all_GPT_outputs.txt')
        # else:
        try:
            incontext_prompt_file = 'LLM_files/LLM_prompts_'+ str(args.num_agents) + '_agents_rand.txt'
            incontext_output_file = 'LLM_files/LLM_output_'+ str(args.num_agents) + '_agents_rand.txt'
            print('using prompts from: ', incontext_prompt_file, incontext_output_file)
            with open(incontext_prompt_file, "r") as f:
                incontext_prompt = f.read()
            with open(incontext_output_file, "r") as f:
                incontext_output = f.read()
        except:
            print('No specific incontext prompts found, using random prompts')
            incontext_prompt_file = os.path.join(args.path, 'LLM_incontext_files/LLM_all_GPT_prompts.txt')
            incontext_output_file = os.path.join(args.path, 'LLM_incontext_files/LLM_all_GPT_outputs.txt')
            with open(incontext_prompt_file, "r") as f:
                incontext_prompt = f.read()
            with open(incontext_output_file, "r") as f:
                incontext_output = f.read()
                
        incontext_prompts = incontext_prompt.split("\n")
        
        incontext_outputs = incontext_output.split("\n")
        len_incontext = len(incontext_prompts)
        if len_incontext < args.num_incontext_prompts:
            print('Not enough incontext prompts, using all available prompts')
            args.num_incontext_prompts = len_incontext
        add_prompt = 'Next sequence of Example prompts and their correct output are from various environments.'
        add_prompt = {"role": "user", "content": add_prompt}
        prompts.append(add_prompt)
        # incontext_prompts = incontext_prompts[:args.num_incontext_prompts]
        # incontext_outputs = incontext_outputs[:args.num_incontext_prompts]
        indices = np.random.choice(len(incontext_prompts), args.num_incontext_prompts, replace=False)
        for i in range(args.num_incontext_prompts):
            ind = indices[i]
            new_prompt = {"role": "user", "content": incontext_prompts[ind]}
            new_ouput = {"role": "assistant", "content": incontext_outputs[ind]}
            prompts.append(new_prompt)
            prompts.append(new_ouput)
        # args.num_incontext_prompts = 0
    if args.leader_model == 'both':
        gpt_num = 2
    else:
        gpt_num = 1
    
    np.random.seed(args.seed)
    # load config
    if not args.u_ref and args.path is not None:
        with open(os.path.join(args.path, "config.yaml"), "r") as f:
            config = yaml.load(f, Loader=yaml.UnsafeLoader)

    # create environments
    num_agents = config.num_agents if args.num_agents is None else args.num_agents
    if (args.preset_scene == 'box' or args.preset_scene =='rand box' or args.preset_scene == 'original box') and args.preset_reset:
        num_agents = 5
        args.area_size = 4.5
    if args.preset_scene == 'corners' and args.preset_reset:
        num_agents = 24
        args.area_size = 6
    env = make_env(
        env_id=config.env if args.env is None else args.env,
        num_agents=num_agents,
        num_obs=args.obs,
        area_size=args.area_size,
        max_step=args.max_step,
        max_travel=args.max_travel,
        use_connect=config.use_connect if not args.u_ref else False,
        reconfig_connect=args.reconfig_connect if not args.u_ref else False,
        use_leader=args.use_leader if not args.u_ref else False,
        preset_reset=args.preset_reset, 
        preset_scene=args.preset_scene,
    )

    if not args.u_ref:
        if args.path is not None:
            path = args.path
            model_path = os.path.join(path, "models")
            if args.step is None:
                models = os.listdir(model_path)
                step = max([int(model) for model in models if model.isdigit()])
            else:
                step = args.step
            print("step: ", step)
            # check if config as dim_factor
            if "dim_factor" not in config:
                dim_factor = 2
            else:
                dim_factor = config.dim_factor
            algo = make_algo(
                algo=config.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                gnn_layers=config.gnn_layers,
                batch_size=config.batch_size,
                buffer_size=config.buffer_size,
                horizon=config.horizon,
                lr_actor=config.lr_actor,
                lr_cbf=config.lr_cbf,
                alpha=config.alpha,
                eps=0.02,
                inner_epoch=8,
                loss_action_coef=config.loss_action_coef,
                loss_unsafe_coef=config.loss_unsafe_coef,
                loss_safe_coef=config.loss_safe_coef,
                loss_h_dot_coef=config.loss_h_dot_coef,
                max_grad_norm=2.0,
                seed=config.seed,
                use_connect=config.use_connect,
                dim_factor=dim_factor,
            )

            algo.load(model_path, step)
            act_fn = jax.jit(algo.act)
        else:
            algo = make_algo(
                algo=args.algo,
                env=env,
                node_dim=env.node_dim,
                edge_dim=env.edge_dim,
                state_dim=env.state_dim,
                action_dim=env.action_dim,
                n_agents=env.num_agents,
                alpha=args.alpha,
            )
            act_fn = jax.jit(algo.act)
            path = os.path.join(f"./logs/{args.env}/{args.algo}")
            if not os.path.exists(path):
                os.makedirs(path)
            step = None
    else:
        assert args.env is not None
        path = os.path.join(f"./logs/{args.env}/nominal")
        if not os.path.exists("./logs"):
            os.mkdir("./logs")
        if not os.path.exists(os.path.join("./logs", args.env)):
            os.mkdir(os.path.join("./logs", args.env))
        if not os.path.exists(path):
            os.mkdir(path)
        algo = None
        act_fn = jax.jit(env.u_ref)
        step = 0

    
    test_key = jr.PRNGKey(args.seed)
    test_keys = jr.split(test_key, 1_000)[: args.epi]
    test_keys = test_keys[args.offset:]

    algo_is_cbf = isinstance(algo, (CentralizedCBF, DecShareCBF))

    if args.cbf is not None:
        assert isinstance(algo, GCBF) or isinstance(algo, GCBFPlus) or isinstance(algo, CentralizedCBF)
        get_bb_cbf_fn_ = ft.partial(get_bb_cbf, algo.get_cbf, env, agent_id=args.cbf, x_dim=0, y_dim=1)
        get_bb_cbf_fn_ = jax_jit_np(get_bb_cbf_fn_)

        def get_bb_cbf_fn(T_graph: GraphsTuple):
            T = len(T_graph.states)
            outs = [get_bb_cbf_fn_(tree_index(T_graph, kk)) for kk in range(T)]
            Tb_x, Tb_y, Tbb_h = jtu.tree_map(lambda *x: jnp.stack(list(x), axis=0), *outs)
            return Tb_x, Tb_y, Tbb_h
    else:
        get_bb_cbf_fn = None
        cbf_fn = None

    for gpt_iter in range(gpt_num):
        if gpt_iter == 0:
            if args.leader_model == 'gpt3.5':
                gpt_model = "gpt-3.5-turbo-1106"
            elif args.leader_model == 'gpt4':
                gpt_model = "gpt-4-1106-preview"
            else:
                if args.leader_model == 'none' or args.leader_model == 'hand':
                    gpt_model = None
                elif args.leader_model == 'fixed':
                    gpt_model = "fixed"
                elif args.leader_model == 'random':
                    gpt_model = "random"
                else:
                    gpt_model = "gpt-3.5-turbo-1106"
        else:
            gpt_model = "gpt-4-1106-preview"

        if args.nojit_rollout:
            print("Only jit step, no jit rollout!")
            rollout_fn = env.rollout_fn_jitstep(act_fn, args.max_step, noedge=True, nograph=args.no_video,
                                                keep_mode=args.keep_mode, prompts=prompts,
                                                num_runtime_incontext_prompts=args.num_runtime_incontext_prompts,
                                                log_dir=log_dir,
                                                leader_model=gpt_model,
                                                LLM_calls=args.num_LLM_calls)

            is_unsafe_fn = None
            is_finish_fn = None
            is_disconnect_fn = None
        else:
            print("jit rollout!")
            rollout_fn = jax_jit_np(env.rollout_fn(act_fn, args.max_step, keep_mode=args.keep_mode))

            is_unsafe_fn = jax_jit_np(jax_vmap(env.collision_mask))
            is_finish_fn = jax_jit_np(jax_vmap(env.finish_mask))
            is_disconnect_fn = jax_jit_np(jax.vmap(env.disconnect_mask))

        rewards = []
        costs = []
        rollouts = []
        is_unsafes = []
        is_finishes = []
        is_disconnects = []
        rates = []
        cbfs = []
        for i_epi in range(args.epi):
            if args.particular_epi != -1 and i_epi != args.particular_epi:
                # print('skipping epi:', i_epi)
                continue
            print('running epi: ', i_epi)
            key_x0, _ = jr.split(test_keys[i_epi], 2)

            if args.nojit_rollout:
                rollout: RolloutResult
                rollout, is_unsafe, is_finish, is_disconnect, stat_output = rollout_fn(key_x0)
                # if not jnp.isnan(rollout.T_reward).any():
                is_unsafes.append(is_unsafe)
                is_finishes.append(is_finish)
                is_disconnects.append(is_disconnect)
                if gpt_model == 'gpt-3.5-turbo-1106':
                    algo_name='GPT3.5'
                elif gpt_model == 'gpt-4-1106-preview':
                    algo_name='GPT4'
                elif gpt_model == 'fixed':
                    algo_name='Fixed leader'
                elif gpt_model == 'random':
                    algo_name='Random leader'
                else:
                    if args.use_leader:
                        algo_name='Hand-designed leader'
                    else:
                        algo_name='GCBF+'
                mean_init_distance, final_mean_distance, num_LLM_calls, dist_traveled, kk, sent_token_count, received_token_count, finish_count, fin_rate, avg_speed_100 = stat_output
                with open(stat_file, "a") as f:
                    # f.write('GPT model, Mean init distance, num_incontext, num_runtime_incontext, num_agents, num_obstacles, finish_mean, Num LLM calls, Distance traveled, Time taken, Tokens sent, Tokens received, Mean final distance \n')
                    f.write(f'{algo_name}, {mean_init_distance:.2f}, {args.num_incontext_prompts}, {args.num_runtime_incontext_prompts}, {args.num_agents}, {args.obs}, {finish_count}, {num_LLM_calls}, {dist_traveled:.2f}, {kk}, {sent_token_count}, {received_token_count}, {final_mean_distance:.2f}, {fin_rate:.2f}, {avg_speed_100:.2f}, {args.keep_mode}, {i_epi} \n')
            else:
                rollout: RolloutResult = rollout_fn(key_x0)
                # if not jnp.isnan(rollout.T_reward).any():
                is_unsafes.append(is_unsafe_fn(rollout.Tp1_graph))
                is_finishes.append(is_finish_fn(rollout.Tp1_graph))
                is_disconnects.append(is_disconnect_fn(rollout.Tp1_graph))

            epi_reward = rollout.T_reward.sum()
            epi_cost = rollout.T_cost.sum()
            rewards.append(epi_reward)
            costs.append(epi_cost)
            rollouts.append(rollout)

            if args.cbf is not None:
                cbfs.append(get_bb_cbf_fn(rollout.Tp1_graph))
            else:
                cbfs.append(None)
            if len(is_unsafes) == 0:
                continue
            safe_rate = 1 - is_unsafes[-1].max(axis=0).mean()
            finish_rate = is_finishes[-1].max(axis=0).mean()
            disconnect_rate = is_disconnects[-1].max(axis=0).mean()

            success_rate = ((1 - is_unsafes[-1].max(axis=0)) * is_finishes[-1].max(axis=0)).mean()
            print(f"epi: {i_epi}, reward: {epi_reward:.3f}, cost: {epi_cost:.3f}, "
                    f"safe rate: {safe_rate * 100:.3f}%,"
                    f"finish rate: {finish_rate * 100:.3f}%, "
                    f"success rate: {success_rate * 100:.3f}%, "
                    f"disconnect rate: {disconnect_rate * 100:.3f}%")

            rates.append(np.array([safe_rate, finish_rate, success_rate]))
        is_unsafe = np.zeros(args.epi)
        is_finish = np.zeros(args.epi)
        is_disconnect = np.zeros(args.epi)
        if args.particular_epi == -1:
            for i in range(args.epi):
                is_unsafe[i] = is_unsafes[i].max()
                is_finish[i] = is_finishes[i].max()
                is_disconnect[i] = is_disconnects[i].max()
        else:
            is_unsafe = np.max(np.stack(is_unsafes), axis=1)
            is_finish = np.max(np.stack(is_finishes), axis=1)
            is_disconnect = np.max(np.stack(is_disconnects), axis=1)
        disconnect_mean, disconnect_std = is_disconnect.mean(), is_disconnect.std()

        safe_mean, safe_std = (1 - is_unsafe).mean(), (1 - is_unsafe).std()
        finish_mean, finish_std = is_finish.mean(), is_finish.std()
        success_mean, success_std = ((1 - is_unsafe) * is_finish).mean(), ((1 - is_unsafe) * is_finish).std()

        print(
            f"reward: {np.mean(rewards):.3f}, min/max reward: {np.min(rewards):.3f}/{np.max(rewards):.3f}, "
            f"cost: {np.mean(costs):.3f}, min/max cost: {np.min(costs):.3f}/{np.max(costs):.3f}, "
            f"safe_rate: {safe_mean * 100:.3f}%, "
            f"finish_rate: {finish_mean * 100:.3f}%, "
            f"success_rate: {success_mean * 100:.3f}%, "
            f"disconnect_rate: {disconnect_mean * 100:.3f}%"
        )

        # save results
        if args.log:
            with open(os.path.join(path, "test_log.csv"), "a") as f:
                f.write(f"{env.num_agents},{args.epi},{env.max_episode_steps},"
                        f"{env.area_size},{env.params['n_obs']},"
                        f"{safe_mean * 100:.3f},{safe_std * 100:.3f},"
                        f"{finish_mean * 100:.3f},{finish_std * 100:.3f},"
                        f"{disconnect_mean * 100:.3f},{disconnect_std * 100:.3f},"
                        f"{success_mean * 100:.3f},{success_std * 100:.3f}\n")

        # save trajectories
        if args.save_traj:
            traj_path = os.path.join(path, "trajs")
            for i in range(args.epi):
                leader_id, leader_dir = jax_vmap(get_leader_id_dir)(rollouts[i].Tp1_graph)
                traj_log = TrajLog(rollouts[i].Tp1_graph.to_tuple(), rollouts[i].T_action, leader_id, leader_dir)
                with open(os.path.join(traj_path, f"traj_log_{i}.pkl"), "wb") as f:
                    pickle.dump(traj_log, f)

        # make video
        if args.no_video:
            continue
        
        videos_dir = pathlib.Path(path) / "videos"
        videos_dir.mkdir(exist_ok=True, parents=True)
        
        # videos_dir.mkdir(exist_ok=True, parents=True)
        for ii, (rollout, Ta_is_unsafe, cbf) in enumerate(zip(rollouts, is_unsafes, cbfs)):
            if algo_is_cbf:
                safe_rate, finish_rate, success_rate = rates[ii] * 100
                video_name = f"n{num_agents}_epi{ii:02}_sr{safe_rate:.0f}_fr{finish_rate:.0f}_sr{success_rate:.0f}"
            else:
                if args.use_llm_leader:
                    video_name = f"LLM_n{num_agents}_step{step}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}"
                else:
                    video_name = f"n{num_agents}_obs_{args.obs}_step{step}_epi{ii:02}_reward{rewards[ii]:.3f}_cost{costs[ii]:.3f}"
            viz_opts = {}
            if args.cbf is not None:
                video_name += f"_cbf{args.cbf}"
                viz_opts["cbf"] = [*cbf, args.cbf]

            if args.use_llm_leader:    
                video_path = os.path.join(log_dir, f"{stamp_str}_{video_name}_{gpt_model}.mp4")
                env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)
                print('Video saved at:', video_path)
            else:
                video_path = videos_dir / f"{stamp_str}_{video_name}.mp4"
                env.render_video(rollout, video_path, Ta_is_unsafe, viz_opts, dpi=args.dpi)
                print('Video saved at:', video_path)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num-agents", type=int, default=None)
    parser.add_argument("--obs", type=int, default=0)
    parser.add_argument("--area-size", type=float, required=True)
    parser.add_argument("--max-step", type=int, default=None)
    parser.add_argument("--path", type=str, default=None)
    parser.add_argument("--n-rays", type=int, default=32)
    parser.add_argument("--alpha", type=float, default=1.0)
    parser.add_argument("--max-travel", type=float, default=None)
    parser.add_argument("--cbf", type=int, default=None)

    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--debug", action="store_true", default=False)
    parser.add_argument("--cpu", action="store_true", default=False)
    parser.add_argument("--u-ref", action="store_true", default=False)
    parser.add_argument("--env", type=str, default=None)
    parser.add_argument("--algo", type=str, default=None)
    parser.add_argument("--step", type=int, default=None)
    parser.add_argument("--epi", type=int, default=5)
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--no-video", action="store_true", default=False)
    parser.add_argument("--nojit-rollout", action="store_true", default=False)
    parser.add_argument("--log", action="store_true", default=False)
    parser.add_argument("--dpi", type=int, default=100)
    parser.add_argument("--reconfig_connect", action="store_true", default=False)
    parser.add_argument("--use_leader", action="store_true", default=False)
    parser.add_argument("--preset_reset", action="store_true", default=False)
    parser.add_argument("--preset_scene", type=str, default=None)
    parser.add_argument("--save-traj", action="store_true", default=False)
    parser.add_argument("--keep-mode", type=int, default=1)
    parser.add_argument("--use-llm-leader", action="store_true", default=False)
    parser.add_argument("--num-incontext-prompts", type=int, default=0)
    parser.add_argument("--num-runtime-incontext-prompts", type=int, default=0)
    parser.add_argument("--leader_model", type=str,choices=['gpt3.5', 'gpt4', 'both', 'none', 'fixed', 'random', 'hand'], default='none')
    parser.add_argument("--particular_epi", type=int, default=-1)
    parser.add_argument("--num_LLM_calls", type=int, default=1)

    args = parser.parse_args()
    test(args)


if __name__ == "__main__":
    with ipdb.launch_ipdb_on_exception():
        main()

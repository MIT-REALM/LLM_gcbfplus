import functools as ft
import pickle

import numpy as np
import pathlib
import jax
import jax.lax as lax
import jax.numpy as jnp
import tqdm
import jax.tree_util as jtu
from time import sleep
from abc import ABC, abstractmethod, abstractproperty
from typing import Callable, NamedTuple, Optional, Tuple
from equinox.debug import breakpoint_if

from ..utils.graph import GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, PRNGKey, Reward, State
from ..utils.utils import jax2np, jax_jit_np, tree_concat_at_front, tree_stack, tree_index, tree_where
from ..utils.LLM_utils import leader_graph

class StepResult(NamedTuple):
    graph: GraphsTuple
    reward: Reward
    cost: Cost
    done: Done
    info: Info


class RolloutResult(NamedTuple):
    Tp1_graph: GraphsTuple
    T_action: Action
    T_reward: Reward
    T_cost: Cost
    T_done: Done
    T_info: Info


class MultiAgentEnv(ABC):

    PARAMS = {}

    def __init__(
            self,
            num_agents: int,
            area_size: float,
            max_step: int = 256,
            max_travel: float = None,
            dt: float = 0.03,
            params: dict = None,
            use_connect: bool = False,
            reconfig_connect: bool = False,
            use_leader: bool = False,
            leader_mode: bool = False,
            prev_leader_mode: bool = False,
            # use_llm: bool = False,
            preset_reset: bool = False,
            preset_scene: str = None,
    ):
        super(MultiAgentEnv, self).__init__()
        self._num_agents = num_agents
        self._dt = dt
        if params is None:
            params = self.PARAMS
        self._params = params
        self._t = 0
        self._max_step = max_step
        self._max_travel = max_travel
        self._area_size = area_size
        self._reconfig_connect = reconfig_connect
        self._use_connect = use_connect
        self._use_leader = use_leader
        self._leader_mode = leader_mode
        self._prev_leader_mode = prev_leader_mode
        # self._use_llm = use_llm

    @property
    def params(self) -> dict:
        return self._params

    @property
    def num_agents(self) -> int:
        return self._num_agents

    @property
    def max_travel(self) -> float:
        return self._max_travel

    @property
    def area_size(self) -> float:
        return self._area_size

    @property
    def dt(self) -> float:
        return self._dt

    @property
    def max_episode_steps(self) -> int:
        return self._max_step

    def clip_state(self, state: State) -> State:
        lower_limit, upper_limit = self.state_lim(state)
        return jnp.clip(state, lower_limit, upper_limit)

    def clip_action(self, action: Action) -> Action:
        lower_limit, upper_limit = self.action_lim()
        return jnp.clip(action, lower_limit, upper_limit)

    @abstractproperty
    def state_dim(self) -> int:
        pass

    @abstractproperty
    def node_dim(self) -> int:
        pass

    @abstractproperty
    def edge_dim(self) -> int:
        pass

    @abstractproperty
    def action_dim(self) -> int:
        pass
    
    @abstractmethod
    def leader_follower_assign(self, graph: GraphsTuple, leader: Array, goal: Array = None) -> GraphsTuple:
        pass

    @abstractmethod
    def reset(self, key: Array) -> GraphsTuple:
        pass

    def reset_np(self, key: Array) -> GraphsTuple:
        """Reset, but without the constraint that it has to be jittable."""
        return self.reset(key)

    @abstractmethod
    def step(self, graph: GraphsTuple, action: Action, get_eval_info: bool = False) -> StepResult:
        pass

    @abstractmethod
    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[State, State],
            limits of the state
        """
        pass

    @abstractmethod
    def action_lim(self) -> Tuple[Action, Action]:
        """
        Returns
        -------
        lower_limit, upper_limit: Tuple[Action, Action],
            limits of the action
        """
        pass

    @abstractmethod
    def control_affine_dyn(self, state: State) -> [Array, Array]:
        pass

    @abstractmethod
    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        pass

    @abstractmethod
    def get_graph(self, state: State) -> GraphsTuple:
        pass

    @abstractmethod
    def u_ref(self, graph: GraphsTuple) -> Action:
        pass

    @abstractmethod
    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        pass

    @abstractmethod
    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        pass

    @abstractmethod
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        pass

    @abstractmethod
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_test_mask(self, graph: GraphsTuple) -> Array:
        pass

    @abstractmethod
    @ft.partial(jax.jit, static_argnums=(0,))
    def disconnect_mask(self, graph: GraphsTuple) -> Array:
        pass

    @abstractmethod
    def collision_mask(self, graph: GraphsTuple) -> Array:
        pass
    


    def multi_control(
        self,
        actor: Callable,
        graph: GraphsTuple,
    ):
        """
        Get all possible actions and corresponding rewards.

        Parameters
        ----------
        env: MultiAgentEnv
        actor: Callable, [GraphsTuple, PRNGKey] -> [Action, LogPi]

        Returns
        -------
        max reward actions and log_pis
        """

        all_graphs, eig_Ls = self.get_all_graphs(graph)
        get_actions = jax.vmap(actor)
        # get_actions = jax.vmap(get_actions, in_axes=(0, None))
        actions = get_actions(all_graphs)
        get_data = jax.vmap(self.step_multi, in_axes=(None, 0))
        rewards = get_data(graph, actions)
        eig_Ls = jnp.asarray(eig_Ls)
        rewards = jnp.asarray(rewards)
        
        scaled_rewards = jnp.where(eig_Ls > 0.1, rewards * eig_Ls, -1e1)
        max_reward_ind = jnp.argmax(scaled_rewards)

        max_reward_graph = tree_index(all_graphs, max_reward_ind)
        # max_reward_action = tree_index(actions, max_reward_ind)
        # max_reward_log_pi = tree_index(log_pis, max_reward_ind)
        return max_reward_graph
    
    def find_leader(self, graph: GraphsTuple) -> Array:
        """
        Find the agent closest to the goal position and assign it as the leader.

        Parameters
        ----------
        graph: GraphsTuple

        Returns
        -------
        leader: int
            index of the leader agent
        
        """
        goal_pos = graph.xg
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        closest_agent = jnp.argmin(jnp.linalg.norm(goal_pos - agent_pos, axis=-1))
        leader = closest_agent.astype(int)
        return leader

    def reset_graph(self, leader_graph: GraphsTuple) -> GraphsTuple:
        """
        Reset the leader follower graph to the goal reaching graph.

        Parameters
        ----------
        leader_graph: GraphsTuple

        Returns
        -------
        graph: GraphsTuple
            the goal reaching graph
        """
        pass

    def best_leader_graph(self, graph: GraphsTuple, actor: Callable) -> GraphsTuple:
        """
        Get the best leader graph.

        Parameters
        ----------
        graph: GraphsTuple
        actor: Callable, [GraphsTuple, PRNGKey] -> [Action, LogPi]

        Returns
        -------
        graph: GraphsTuple
            the best leader graph
        """
        leaders = jnp.arange(self.num_agents)  # all agents are potential leaders
        goal_pos_rel = jnp.array([[0.0, 1.0], [0.0, -1.0], [1.0, 0.0], [-1.0, 0.0]])  # go up, down, right, left
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        real_goal_pos = graph.xg

        def get_goal_pos(pos):
            return pos + goal_pos_rel

        goal_pos = jax.vmap(get_goal_pos)(agent_pos)  # (n_agents, 4, 2)

        assign_fn = ft.partial(self.leader_follower_assign, graph)

        def assign_one_leader(leader_id, goals):
            return jax.vmap(ft.partial(assign_fn, leader_id))(goals)

        leader_graphs = jax.vmap(assign_one_leader)(leaders, goal_pos)  # (n_leader, 4, ...)

        # actions should be large enough
        actions = jax.vmap(jax.vmap(actor))(leader_graphs)  # (n_leader, 4, n_agents, action_dim)
        actions_mean = jnp.linalg.norm(actions, axis=-1).mean(axis=-1)  # (n_leader, 4)
        # large_enough_action = actions_mean > 0.2
        # best_action_id = jnp.unravel_index(jnp.argmax(actions_mean), actions_mean.shape)

        # the system should head towards the goal
        goal_dir = ((real_goal_pos - agent_pos) /
                    (jnp.linalg.norm(real_goal_pos - agent_pos, axis=-1, keepdims=True) + 1e-6))  # (n_agents, 2)
        average_goal_dir = jnp.mean(goal_dir, axis=0)[None, None, :]   # (1, 1, 2)
        actions_mean_dir = jnp.mean(actions, axis=2)    # (n_leader, 4, action_dim)
        dot_prod = jnp.sum(average_goal_dir * actions_mean_dir, axis=-1)  # (n_leader, 4)

        score = actions_mean + dot_prod
        # dot_prod = jnp.where(large_enough_action, dot_prod, dot_prod - 1e2)
        best_action_id = jnp.unravel_index(jnp.argmax(score), score.shape)

        # jax.debug.breakpoint()
        # breakpoint()

        return jtu.tree_map(lambda x: x[best_action_id], leader_graphs)


    def leader_control(
            self,
            actor: Callable,
            graph: GraphsTuple,
            t_mode: Array,
            keep_mode: int,
            leader_id: Array = None,
            leader_goal: Array = None
    ) -> [GraphsTuple, Array]:
        """
        Check if the average speed is below a threshold and assign the leader-follower relationship.
        Otherwise, use the original graph.

        Parameters
        ----------
        actor: Callable, [GraphsTuple, PRNGKey] -> [Action, LogPi]
        graph: GraphsTuple
        t_mode: Array,

        Returns
        -------
        new graph with leader-follower assignment if the average speed is below the threshold
        original graph if the average speed is above the threshold
        """
        # judge if the current graph is in the leader mode
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        in_leader_mode = ~jnp.allclose(goals, graph.xg)
        cur_leader = jnp.argmax(jnp.linalg.norm(goals - agent_states, axis=-1))
        leader_graph_keep = self.leader_follower_assign(graph, cur_leader, goals[cur_leader])

        # construct the goal graph
        goal_graph = tree_where(in_leader_mode, self.reset_graph(graph), graph)

        # construct the leader graph
        # leader_graph = self.leader_follower_assign(graph, leader=self.find_leader(graph))
        if leader_id is None:
            leader_graph = self.best_leader_graph(graph, actor)
        else:
            leader_graph = self.leader_follower_assign(graph, leader_id, leader_goal + agent_states[leader_id])

        # judge of we want to use the leader follower mode1
        actions = actor(graph)
        action_norm = jnp.linalg.norm(actions, axis=-1)
        low_speed = jnp.mean(action_norm) < 0.2
        far_from_goal = jnp.linalg.norm(graph.xg - agent_states[:, :2], axis=-1).mean() > 0.5
        use_leader_mode = low_speed & far_from_goal

        # get the desired graph
        desired_graph = tree_where(use_leader_mode, leader_graph, graph)
        desired_graph = tree_where((~use_leader_mode & in_leader_mode), goal_graph, desired_graph)
        desired_graph = tree_where(((t_mode < keep_mode) & in_leader_mode), leader_graph_keep, desired_graph)
        desired_graph = tree_where(use_leader_mode, leader_graph, desired_graph)

        t_mode += 1
        t_mode = jnp.where(use_leader_mode & ~in_leader_mode, jnp.zeros_like(t_mode), t_mode)
        # t_mode = jnp.where((~use_leader_mode & (t_mode >= keep_mode)), jnp.zeros_like(t_mode), t_mode)

        return desired_graph, t_mode

    def rollout_fn(self, policy: Callable, rollout_length: int = None, keep_mode: int = 10) -> Callable[[PRNGKey], RolloutResult]:
        rollout_length = rollout_length or self.max_episode_steps

        def body(inp, _):
            graph, t_mode = inp

            if self._reconfig_connect:
                graph = self.multi_control(policy, graph)

            if self._use_leader:
                graph, t_mode = self.leader_control(policy, graph, t_mode, keep_mode)

            action = policy(graph)
            graph_new, reward, cost, done, info = self.step(graph, action, get_eval_info=True)
            return (graph_new, t_mode), (graph_new, action, reward, cost, done, info)

        def fn(key: PRNGKey) -> RolloutResult:
            graph0 = self.reset(key)
            (graph_final, t_mode_final), (T_graph, T_action, T_reward, T_cost, T_done, T_info) = lax.scan(
                body, (graph0, jnp.zeros(())), None, length=rollout_length
            )
            Tp1_graph = tree_concat_at_front(graph0, T_graph, axis=0)

            return RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info)

        return fn

    def rollout_fn_jitstep(
            self,
            policy: Callable,
            rollout_length: int = None,
            noedge: bool = False,
            nograph: bool = False,
            keep_mode: int = 10,
            prompts = None,
            num_runtime_incontext_prompts: int = 0,
            log_dir=None,
            leader_model=None,
            LLM_calls: int = 1,
    ):
        rollout_length = rollout_length or self.max_episode_steps
        
        jit_policy = jax.jit(policy)
        jit_step = jax.jit(ft.partial(self.step, get_eval_info=False))
        jit_leader_follower_assign = jax.jit(self.leader_follower_assign)
        
        is_unsafe_fn = jax_jit_np(self.unsafe_test_mask)
        is_finish_fn = jax_jit_np(self.finish_mask)
        is_disconnect_fn = jax_jit_np(self.disconnect_mask)


        def fn(key: PRNGKey) -> [RolloutResult, Array, Array]:
            graph0 = self.reset_np(key)
            graph = graph0
            T_output = []
            is_unsafes = []
            is_finishes = []
            is_disconnects = []

            is_unsafes.append(is_unsafe_fn(graph0))
            is_finishes.append(is_finish_fn(graph0))
            is_disconnects.append(is_disconnect_fn(graph0))

            graph0 = jax2np(graph0)
            leader_assign_count = 0
            prompt_time_gap = 0
            t_mode = jnp.zeros(())
            prompt = np.array(prompts).copy()
            prompt = prompt.tolist()
            all_prompts = np.array(prompt).copy()
            all_prompts = all_prompts.tolist()
            action = None
            with open(f"{log_dir}/log_message.txt", "a") as f:
                f.write('Using GPT model:')
                f.write(str(leader_model))
                f.write('\n')
                for p in all_prompts:
                    f.write(str(p))
                    f.write('\n')
            
            num_LLM_calls = 0
            # prev_state = None
            dist_traveled = 0
            sent_token_count = 0
            received_token_count = 0
            mean_init_distance = jnp.linalg.norm(graph.xg - graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2], axis=-1).mean()
            mean_speed = []
            for kk in tqdm.trange(rollout_length, ncols=80):
                # judge if leader is needed
                
                if self._use_leader:
                    graph, num_LLM_calls, prompt, all_prompts, leader_assign_count, t_mode, prompt_time_gap, sent_token_count, received_token_count = leader_graph(graph, self._num_agents, jit_policy, keep_mode, jit_leader_follower_assign, leader_model, kk ,log_dir, num_runtime_incontext_prompts, self.leader_control, policy,prompt,all_prompts,LLM_calls,self.reset_graph, leader_assign_count, t_mode, prompt_time_gap, num_LLM_calls, sent_token_count, received_token_count)
                       
                # step environment
                action = jit_policy(graph)
                avg_speed = jnp.linalg.norm(action, axis=-1).mean()
                mean_speed.append(avg_speed)
                prev_states = graph.type_states(type_idx=0, n_type=self.num_agents)
                graph, reward, cost, done, info = jit_step(graph, action)
                new_states = graph.type_states(type_idx=0, n_type=self.num_agents)
                dist_traveled += jnp.linalg.norm(new_states - prev_states, axis=-1).mean()
                output = (graph, action, reward, cost, done, info)

                is_unsafes.append(is_unsafe_fn(graph))
                is_finishes.append(is_finish_fn(graph))
                is_disconnects.append(is_disconnect_fn(graph))

                output = jax2np(output)
                if noedge:
                    output = (output[0].without_edge(), *output[1:])
                if nograph:
                    output = (None, *output[1:])
                T_output.append(output)
                final_distance = jnp.linalg.norm(graph.xg - graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2], axis=-1)
                finish_count = final_distance < 0.2
                final_mean_distance = jnp.mean(final_distance)
                finish_count = sum(finish_count.astype(int))
                if finish_count == self._num_agents:
                    break
            # if stat_file is not None:
                # Log following in the stat file
                # f.write('GPT model, num_incontext, num_runtime_incontext, num_agents, finish_mean, Num LLM calls, Distance traveled, Time taken, Tokens sent, Tokens received \n')
            # see if there's any obstacle in path to goal
            # sample points from current states to goal on the straight line
            c_ = jnp.linspace(0, 1, 20)
            c_ = c_[..., None, None]
            agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
            points = (1 - c_) * agent_states[None,...] + c_ * graph.xg[None,...]
            
            # generate graphs for each of the points
                
            currect_connectivity = graph.connectivity
            current_goals = graph.xg
            current_obs = graph.env_states.obstacle
            env_state_fn = ft.partial(self.EnvState, obstacle=current_obs, goal=current_goals)

            env_states = jax.vmap(env_state_fn)(points)
            
            graphs_for_points_fn = ft.partial(self.get_graph, adjacency=currect_connectivity, goals=current_goals)
            graphs_for_points = jax.vmap(graphs_for_points_fn)(env_states)
            # get the unsafe masks for each of the points
            unsafe_masks = jax.vmap(self.unsafe_test_mask)(graphs_for_points)
            # if there's any unsafe mask, then we need to use the leader mode
            unsafe_mode = jnp.any(unsafe_masks, axis=0)

            if not unsafe_mode.any():
                finish_count = self._num_agents

            # find average speed of last 100 steps
            avg_speed_last_100 = jnp.mean(jnp.array(mean_speed[-100:]))
            
            finish = is_finishes[-1].astype(int).sum()

            stat_output = mean_init_distance, final_mean_distance, num_LLM_calls, dist_traveled, kk, sent_token_count, received_token_count, finish_count, finish, avg_speed_last_100
            with open(f"{log_dir}/log_message.txt", "a") as f:
                f.write('Number of LLM calls: ')
                f.write(str(num_LLM_calls))
                f.write('\n')
                f.write('Distance traveled: ')
                f.write(str(dist_traveled))
                f.write('\n')
                f.write('Mean init distance: ')
                f.write(str(mean_init_distance))
                f.write('\n')
                f.write('Total time traveled: ')
                f.write(str(kk))
                f.write('\n')
                f.write('Total token sent: ')
                f.write(str(sent_token_count))
                f.write('\n')
                f.write('Total token received: ')
                f.write(str(received_token_count))
                f.write('\n')
                f.write('Final mean distance: ')
                f.write(str(final_mean_distance))
                f.write('\n')

            # Concatenate everything together.

            T_graph = [o[0] for o in T_output]
            if noedge:
                T_graph = [graph0.without_edge()] + T_graph
            else:
                T_graph = [graph0] + T_graph
            del graph0
            T_action = [o[1] for o in T_output]
            T_reward = [o[2] for o in T_output]
            T_cost = [o[3] for o in T_output]
            T_done = [o[4] for o in T_output]
            T_info = [o[5] for o in T_output]
            del T_output

            if nograph:
                T_graph = None
            else:
                T_graph = tree_stack(T_graph)
            T_action = tree_stack(T_action)
            T_reward = tree_stack(T_reward)
            T_cost = tree_stack(T_cost)
            T_done = tree_stack(T_done)
            T_info = tree_stack(T_info)

            Tp1_graph = T_graph

            rollout_result = jax2np(RolloutResult(Tp1_graph, T_action, T_reward, T_cost, T_done, T_info))
            return rollout_result, np.stack(is_unsafes, axis=0), np.stack(is_finishes, axis=0), np.stack(is_disconnects, axis=0), stat_output

        return fn

    @abstractmethod
    def render_video(
        self, rollout: RolloutResult, video_path: pathlib.Path, Ta_is_unsafe=None, viz_opts: dict = None, **kwargs
    ) -> None:
        pass

    @abstractmethod
    def finish_mask(self, graph: GraphsTuple) -> Array:
        pass

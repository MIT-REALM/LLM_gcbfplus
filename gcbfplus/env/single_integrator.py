import functools as ft
import pathlib
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from typing import NamedTuple, Tuple, Optional

from jax.lax import while_loop
from equinox.debug import breakpoint_if

from ..utils.graph import EdgeBlock, GetGraph, GraphsTuple
from ..utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState
from ..utils.utils import merge01, jax_vmap, tree_stack, tree_where
from .base import MultiAgentEnv, RolloutResult
from .obstacle import Obstacle, Rectangle
from .plot import render_video
from .utils import get_lidar, inside_obstacles, lqr, get_node_goal_rng


class SingleIntegrator(MultiAgentEnv):

    AGENT = 0
    GOAL = 1
    OBS = 2

    class EnvState(NamedTuple):
        agent: State
        goal: State
        obstacle: Obstacle

        @property
        def n_agent(self) -> int:
            return self.agent.shape[0]

    EnvGraphsTuple = GraphsTuple[State, EnvState]

    PARAMS = {
        "car_radius": 0.05,
        "comm_radius": 0.5,
        "n_rays": 32,
        "obs_len_range": [0.1, 0.6],
        "n_obs": 8,
    }

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
            preset_reset: bool = False,
            preset_scene: str = None,
    ):
        super(SingleIntegrator, self).__init__(num_agents, area_size, max_step, max_travel, dt, params,use_connect, reconfig_connect, use_leader, leader_mode, prev_leader_mode, preset_reset, preset_scene)
        self._A = np.zeros((self.state_dim, self.state_dim), dtype=np.float32) * self._dt + np.eye(self.state_dim)
        self._B = np.array([[1.0, 0.0], [0.0, 1.0]]) * self._dt
        self._Q = np.eye(self.state_dim) * 2
        self._R = np.eye(self.action_dim)
        self._K = jnp.array(lqr(self._A, self._B, self._Q, self._R))
        self.create_obstacles = jax_vmap(Rectangle.create)
        self.create_obstacles_single = Rectangle.create
        self.use_connect = use_connect
        self.preset_reset = preset_reset
        self.preset_scene = preset_scene

    @property
    def state_dim(self) -> int:
        return 2  # x, y

    @property
    def node_dim(self) -> int:
        return 3  # indicator: agent: 001, goal: 010, obstacle: 100

    @property
    def edge_dim(self) -> int:
        return 4  # x_rel, y_rel, connectivity

    @property
    def action_dim(self) -> int:
        return 2  # vx, vy

    def get_all_graphs(self, graph: GraphsTuple):
        # Get the adjacency matrix
        adj_matrix = graph.connectivity

        def remove_edge(adjacency_matrix, edges):
            # Copy the original adjacency matrix
            # new_adjacency_matrix = np.array(adjacency_matrix)
            
            # Remove all specified edges
            for i, j in edges:
                adjacency_matrix = adjacency_matrix.at[i, j].set(0)
                adjacency_matrix = adjacency_matrix.at[j, i].set(0)
            
            return adjacency_matrix

        def all_possible_matrices(adjacency_matrix):
            num_nodes = adjacency_matrix.shape[0]
            
            # Generate all possible edges
            edges = np.array([(i, j) for i in range(num_nodes) for j in range(i + 1, num_nodes)])
            edges = edges.reshape((-1, 1, 2))  # Add a leading dimension for vmap
            
            # Use vectorized operation to remove edges
            matrices_list = jax.vmap(lambda edges: remove_edge(adjacency_matrix, edges))(edges)
            
            return matrices_list

        # def remove_edge(adjacency_matrix, i, j):
        #     # Copy the original adjacency matrix and remove the edge (i, j)
        #     adjacency_matrix = adjacency_matrix.at[i, j].set(0)
        #     adjacency_matrix = adjacency_matrix.at[j, i].set(0)
        #     return adjacency_matrix 

        # def all_possible_matrices(adjacency_matrix):
        #     num_nodes = adjacency_matrix.shape[0]
            
        #     # Loop through all edges and generate matrices by removing one edge at a time
        #     matrices_list = [adjacency_matrix]
        #     count = []
        #     for i in range(num_nodes):
        #         Ni = adjacency_matrix[i] > 0
        #         Ni = Ni.astype(int)
        #         count_i = 0
        #         for j in Ni:
        #             new_adj = remove_edge(adjacency_matrix, i, j)
        #             # new_D = jnp.diag(jnp.sum(new_adj, axis=1))
        #             # new_L = new_D - new_adj
        #             # eig_Ls,_ = jnp.linalg.eigh(new_L)
        #             # matrices_list = tree_where(eig_Ls[ind_v] > 1e-5, [matrices_list].append(new_adj), matrices_list)
        #             # count_i = jnp.where(eig_Ls[ind_v] > 1e-5, count_i + 1, count_i)
        #             # if eig_Ls[ind_v] > 1e-5:
        #             matrices_list.append(new_adj)
        #             count_i += 1
        #         count.append(count_i)
        #     return matrices_list, count
        
        jit_all_possible_matrices = jax.jit(all_possible_matrices)
        
        adjacencies = jit_all_possible_matrices(adj_matrix)
        adjacencies = jnp.append(adjacencies, adj_matrix.reshape(1, self.num_agents, self.num_agents), axis=0)
        adj_mats = jnp.asarray(adjacencies)
        D_mats = jnp.diag(jnp.sum(adj_mats, axis=1))
        L_mats = D_mats - adj_mats
        eig_Ls,_ = jnp.linalg.eigh(L_mats)
        
        eig_Ls = jnp.sort(eig_Ls, axis=1)[:, 1]
        graphs = []
        
        for adj in adjacencies:
            graphs.append(self.change_adjacency(graph, adj))
        
        return tree_stack(graphs), eig_Ls
    
    def change_adjacency(self, graph: GraphsTuple, adjacency: Array) -> GraphsTuple:
        assert graph.is_single
        edge_feats = graph.edges
        connectivity = adjacency
        connectivity = connectivity[graph.receivers, graph.senders]
        edge_has_obs = graph.senders >= self.num_agents  # this edge has obstacles
        edge_feats = edge_feats.at[:, 2:].set(jnp.concatenate([connectivity[:, None], 1 - connectivity[:, None]], axis=-1))
        edge_feats = jnp.where(~edge_has_obs[:, None], edge_feats, edge_feats.at[:, 2:].set(jnp.zeros((edge_feats.shape[0], 2))))

        return graph._replace(connectivity=adjacency)
    
    def step_multi(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ):
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)

        # compute reward and cost
        reward = jnp.linalg.norm(goals - agent_states, axis=-1).mean()
        dist2goal = jnp.linalg.norm(goals - next_agent_states, axis=-1)
        reward -= dist2goal.mean()
        # reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()  # * 0.01

        assert reward.shape == tuple()

        return reward
    
    def leader_follower_assign(self, graph: GraphsTuple, leader: Array, goal: Array = None) -> GraphsTuple:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_state = graph.xg
        obs = graph.env_states.obstacle

        def body_(i_agent, assign_input):
            """Find the leader of the i-th agent"""
            # agents that are already assigned leaders (set A), agents that are not assigned leaders (set B)
            # assigned_id, not_assgined_id, leader_id = assign_input
            assigned_mask, leader_id = assign_input

            pos_not_assigned = jnp.where(~assigned_mask[:, None], agent_state, 1000000 * jnp.ones((self.num_agents, 2)))
            pos_assigned = jnp.where(assigned_mask[:, None], agent_state, 100 * jnp.ones((self.num_agents, 2)))

            # jax.debug.breakpoint()

            dist = jnp.linalg.norm(pos_not_assigned[None, :] - pos_assigned[:, None], axis=-1)
            # follower_i = jnp.argmin(dist, axis=1)  # this is the follower
            # leader_i = jnp.argmin(dist, axis=0)  # this is the leader
            min_dist_id = jnp.argmin(dist)
            follower_i = min_dist_id % self.num_agents
            leader_i = min_dist_id // self.num_agents

            # breakpoint_if(follower_i == leader_i)

            leader_id = leader_id.at[follower_i].set(leader_i)
            assigned_mask = assigned_mask.at[follower_i].set(True)

            return assigned_mask, leader_id

        init_assigned_mask = (jnp.zeros(self.num_agents)).astype(bool)
        init_assigned_mask = init_assigned_mask.at[leader].set(True)
        init_leader_id = (jnp.ones(self.num_agents) * -1).astype(int)

        assigned_mask, leader_id = jax.lax.fori_loop(0, self.num_agents - 1, body_, (init_assigned_mask, init_leader_id))


        temp_goal = agent_state[leader_id.squeeze()]
        temp_goal = temp_goal.at[leader].set(goal_state[leader] if goal is None else goal)
        # breakpoint()

        
        # define new adjacency based on lead-follow assignment
        new_adj = jnp.zeros((self.num_agents, self.num_agents))
        for i in range(self.num_agents):
            new_adj = new_adj.at[i, leader_id[i]].set(1)
            new_adj = new_adj.at[leader_id[i], i].set(1)
        
        # remove self connections
        new_adj = new_adj.at[jnp.arange(self.num_agents), jnp.arange(self.num_agents)].set(0)
        
        new_adj = new_adj.astype(jnp.bool_)

        # create new graph
        new_graph = self.get_graph(self.EnvState(agent_state, temp_goal, obs), new_adj, graph.xg)
        
        return new_graph

    def reset_graph(self, leader_graph: GraphsTuple) -> GraphsTuple:
        agent_states = leader_graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_state = leader_graph.xg
        obs = leader_graph.env_states.obstacle
        env_state = self.EnvState(agent_states, goal_state, obs)

        # calculate the connectivity based on the current distance
        distances = jnp.linalg.norm(agent_states[:, None, :2] - agent_states[None, :, :2], axis=-1)
        distances = distances + jnp.eye(self.num_agents) * (self._params["comm_radius"] * 2 + 1)
        adjacency = distances < 1. * self._params["comm_radius"]

        return self.get_graph(env_state, adjacency, leader_graph.xg)

    def reset_box_original(self):
        states =jnp.ones((self.num_agents, 2))
        states = states.at[:, 0].set(jnp.array([-2.5, -2.1, -2.5, -2.1, -2.3]))
        states = states.at[:, 1].set(jnp.array([-2.5, -2.5, -2.1, -2.1, -1.9]))
        
        goals = states
        goals = goals.at[:, 0].set(goals[:, 0] + 4.2)
        goals = goals.at[:, 1].set(goals[:, 1] + 3.0)
        #swap goals[0, :] and goals[2, :]
        temp = goals[0, :].copy()
        goals = goals.at[0, :].set(goals[2, :])
        goals = goals.at[2, :].set(temp)

        obs = jnp.ones((14, 2))
        obs_len = jnp.ones((14, 2))
        obs_theta = jnp.zeros((14, 1))

        # first four obstacles for room of size 6 x 6 centered at (0, 0)
        obs = obs.at[:4].set(jnp.array([[-3, 0], [0, 3], [3, 0], [0, -3]]))
        obs_len = obs_len.at[:4].set(jnp.array([[0.2, 6], [6, 0.2], [0.2, 6], [6, 0.2]]))

        # next obstacle at (-1.2, -3) of height 1.5
        obs_new = jnp.array([[-1.2, -2.25]]).reshape(2,)
        obs = obs.at[4, :].set(obs_new)
        obs_len = obs_len.at[4].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        # next obstacle at (1.6, 0) of height 1.2
        obs = obs.at[5].set(jnp.array([[1.6, 0.6]]).reshape(2,))
        obs_len = obs_len.at[5].set(jnp.array([[0.2, 1.2]]).reshape(2,))

        # next obstacle at (1.6, 0) of width 1.4
        obs = obs.at[6].set(jnp.array([[2.3, 0]]).reshape(2,))
        obs_len = obs_len.at[6].set(jnp.array([[1.4, 0.2]]).reshape(2,))

        # next obstacle at (-2, 1.5) of height 1.5
        obs = obs.at[7].set(jnp.array([[-2, 2.25]]).reshape(2,))
        obs_len = obs_len.at[7].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        # next obstacle at (-0.5, 1.5) of height 1.5
        obs = obs.at[8].set(jnp.array([[-0.5, 2.25]]).reshape(2,))
        obs_len = obs_len.at[8].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        # next obstacle at (-2, 1.5) of width 1.5
        obs = obs.at[9].set(jnp.array([[-1.25, 1.5]]).reshape(2,))
        obs_len = obs_len.at[9].set(jnp.array([[1.5, 0.2]]).reshape(2,))

        # next 4 obstacles making a box of size 1 x 1 centered at (0, 0)
        obs = obs.at[10:].set(jnp.array([[-0.5, 0], [0, 0.5], [0.5, 0], [0, -0.5]]))
        obs_len = obs_len.at[10:].set(jnp.array([[0.2, 1], [1, 0.2], [0.2, 1], [1, 0.2]]))
        
        states = states * 0.7
        obs = obs * 0.7
        goals = goals * 0.7
        obs_len = obs_len * 0.7
        states = states + 2.2
        obs = obs + 2.2
        goals = goals + 2.2
        goals = goals.at[:, 0].set(goals[:, 0] + 0.3)

        return states, goals, obs, obs_len, obs_theta


    def reset_box(self, key):
        states =jnp.ones((self.num_agents, 2))
        states = states.at[:, 0].set(jnp.array([-2.5, -2.1, -2.5, -2.1, -2.3]))
        states = states.at[:, 1].set(jnp.array([-2.5, -2.5, -2.1, -2.1, -1.9]))
        
        goals = states
        goals = goals.at[:, 0].set(goals[:, 0] + 4.2)
        goals = goals.at[:, 1].set(goals[:, 1] + 3.0)
        #swap goals[0, :] and goals[2, :]
        temp = goals[0, :].copy()
        goals = goals.at[0, :].set(goals[2, :])
        goals = goals.at[2, :].set(temp)

        obs = jnp.ones((17, 2))
        obs_len = jnp.ones((17, 2))
        obs_theta = jnp.zeros((17, 1))

        # first four obstacles for room of size 6 x 6 centered at (0, 0)
        obs = obs.at[:4].set(jnp.array([[-3, 0], [0, 3], [3, 0], [0, -3]]))
        obs_len = obs_len.at[:4].set(jnp.array([[0.2, 6], [6, 0.2], [0.2, 6], [6, 0.2]]))

        # next obstacle at (-1.2, -3) of height 1.5
        obs_new = jnp.array([[-1.2, -2.25]]).reshape(2,)
        obs = obs.at[4, :].set(obs_new)
        obs_len = obs_len.at[4].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        # next obstacle at (1.6, 0) of height 1.2
        obs = obs.at[5].set(jnp.array([[1.6, 0.6]]).reshape(2,))
        obs_len = obs_len.at[5].set(jnp.array([[0.2, 1.2]]).reshape(2,))

        # next obstacle at (1.6, 0) of width 1.4
        obs = obs.at[6].set(jnp.array([[2.3, 0]]).reshape(2,))
        obs_len = obs_len.at[6].set(jnp.array([[1.4, 0.2]]).reshape(2,))

        # next obstacle at (-2, 1.5) of height 1.5
        obs = obs.at[7].set(jnp.array([[-2, 2.25]]).reshape(2,))
        obs_len = obs_len.at[7].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        # next obstacle at (-0.5, 1.5) of height 1.5
        obs = obs.at[8].set(jnp.array([[-0.5, 2.25]]).reshape(2,))
        obs_len = obs_len.at[8].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        # next obstacle at (-2, 1.5) of width 1.5
        obs = obs.at[9].set(jnp.array([[-1.25, 1.5]]).reshape(2,))
        obs_len = obs_len.at[9].set(jnp.array([[1.5, 0.2]]).reshape(2,))

        # next 4 obstacles making a box of size 1 x 1 centered at (0, 0)
        obs = obs.at[10:14].set(jnp.array([[-0.5, 0], [0, 0.5], [0.5, 0], [0, -0.5]]))
        obs_len = obs_len.at[10:14].set(jnp.array([[0.2, 1], [1, 0.2], [0.2, 1], [1, 0.2]]))
        
        # next obstacle at (1.3, 1.2) of width 0.6
        use_key, key = jr.split(key, 2)
        obs_len_rand = jr.uniform(use_key, (1,), minval=-0.3, maxval=0.3)

        obs_len = obs_len.at[14].set(jnp.array([[0.6, 0.2]]).reshape(2,))
        obs_len = obs_len.at[14, 0].set(obs_len_rand.reshape() + obs_len[14, 0])

        obs = obs.at[14].set(jnp.array([[1.3, 1.2]]).reshape(2,))
        
        # use_key, key = jr.split(key, 2)
        # obs_x_rand = jr.uniform(use_key, (1,), minval=0.0, maxval=0.5)
        # obs = obs.at[14, 0].set(obs[14, 0] + obs_x_rand.reshape())
        
        # next obstacle at (0.5, -1.25) of height 0.5
        obs = obs.at[15].set(jnp.array([[0.5, -1.25]]).reshape(2,))
        obs_len = obs_len.at[15].set(jnp.array([[0.2, 0.5]]).reshape(2,))

        use_key, key = jr.split(key, 2)
        obs_x_rand = jr.uniform(use_key, (1,), minval=-0.5, maxval=0.0)
        obs = obs.at[15, 0].set(obs[15, 0] + obs_x_rand.reshape())

        use_key, key = jr.split(key, 2)
        obs_len_rand = jr.uniform(use_key, (1,), minval=-0.3, maxval=0.3)
        obs_len = obs_len.at[15, 1].set(obs_len_rand.reshape()+obs_len[15, 1])

        # next obstacle at (-1.5, 0) of height 1.5
        obs = obs.at[16].set(jnp.array([[-1.5, 0.0]]).reshape(2,))
        obs_len = obs_len.at[16].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        use_key, key = jr.split(key, 2)
        obs_len_rand = jr.uniform(use_key, (1,), minval=-0.3, maxval=0.3)
        obs_len = obs_len.at[16, 1].set(obs_len_rand.reshape()+obs_len[16, 1])

        use_key, key = jr.split(key, 2)
        obs_x_rand = jr.uniform(use_key, (1,), minval=-0.5, maxval=0.0)
        obs = obs.at[16, 0].set(obs[16, 0] + obs_x_rand.reshape())

        states = states * 0.7   
        obs = obs * 0.7
        goals = goals * 0.7
        obs_len = obs_len * 0.7
        states = states + 2.2
        obs = obs + 2.2
        goals = goals + 2.2
        goals = goals.at[:, 0].set(goals[:, 0] + 0.3)

        return states, goals, obs, obs_len, obs_theta

    # def reset_box(self):
    #     states =jnp.ones((self.num_agents, 2))
    #     states = states.at[:, 0].set(jnp.array([-2.5, -2.1, -2.5, -2.1, -2.3]))
    #     states = states.at[:, 1].set(jnp.array([-2.5, -2.5, -2.1, -2.1, -1.9]))
        
    #     goals = states
    #     goals = goals.at[:, 0].set(goals[:, 0] + 4.2)
    #     goals = goals.at[:, 1].set(goals[:, 1] + 3.0)
    #     #swap goals[0, :] and goals[2, :]
    #     temp = goals[0, :].copy()
    #     goals = goals.at[0, :].set(goals[2, :])
    #     goals = goals.at[2, :].set(temp)

    #     obs = jnp.ones((14, 2))
    #     obs_len = jnp.ones((14, 2))
    #     obs_theta = jnp.zeros((14, 1))

    #     # first four obstacles for room of size 6 x 6 centered at (0, 0)
    #     obs = obs.at[:4].set(jnp.array([[-3, 0], [0, 3], [3, 0], [0, -3]]))
    #     obs_len = obs_len.at[:4].set(jnp.array([[0.2, 6], [6, 0.2], [0.2, 6], [6, 0.2]]))

    #     # next obstacle at (-1.2, -3) of height 1.5
    #     obs_new = jnp.array([[-1.2, -2.25]]).reshape(2,)
    #     obs = obs.at[4, :].set(obs_new)
    #     obs_len = obs_len.at[4].set(jnp.array([[0.2, 1.5]]).reshape(2,))

    #     # next obstacle at (1.6, 0) of height 1.2
    #     obs = obs.at[5].set(jnp.array([[1.6, 0.6]]).reshape(2,))
    #     obs_len = obs_len.at[5].set(jnp.array([[0.2, 1.2]]).reshape(2,))

    #     # next obstacle at (1.6, 0) of width 1.4
    #     obs = obs.at[6].set(jnp.array([[2.3, 0]]).reshape(2,))
    #     obs_len = obs_len.at[6].set(jnp.array([[1.4, 0.2]]).reshape(2,))

    #     # next obstacle at (-2, 1.5) of height 1.5
    #     obs = obs.at[7].set(jnp.array([[-2, 2.25]]).reshape(2,))
    #     obs_len = obs_len.at[7].set(jnp.array([[0.2, 1.5]]).reshape(2,))

    #     # next obstacle at (-0.5, 1.5) of height 1.5
    #     obs = obs.at[8].set(jnp.array([[-0.5, 2.25]]).reshape(2,))
    #     obs_len = obs_len.at[8].set(jnp.array([[0.2, 1.5]]).reshape(2,))

    #     # next obstacle at (-2, 1.5) of width 1.5
    #     obs = obs.at[9].set(jnp.array([[-1.25, 1.5]]).reshape(2,))
    #     obs_len = obs_len.at[9].set(jnp.array([[1.5, 0.2]]).reshape(2,))

    #     # next 4 obstacles making a box of size 1 x 1 centered at (0, 0)
    #     obs = obs.at[10:].set(jnp.array([[-0.5, 0], [0, 0.5], [0.5, 0], [0, -0.5]]))
    #     obs_len = obs_len.at[10:].set(jnp.array([[0.2, 1], [1, 0.2], [0.2, 1], [1, 0.2]]))
        
    #     states = states * 0.7
    #     obs = obs * 0.7
    #     goals = goals * 0.7
    #     obs_len = obs_len * 0.7
    #     states = states + 2.2
    #     obs = obs + 2.2
    #     goals = goals + 2.2
    #     goals = goals.at[:, 0].set(goals[:, 0] + 0.3)

    #     return states, goals, obs, obs_len, obs_theta

    def reset_box_rand(self, key):
        use_key, key = jr.split(key, 2)

        states =jnp.ones((self.num_agents, 2))
        states = states.at[:, 0].set(jnp.array([-2.5, -2.1, -2.5, -2.1, -2.3]))
        states = states.at[:, 1].set(jnp.array([-2.5, -2.5, -2.1, -2.1, -1.9]))
        
        goals = states
        goals = goals.at[:, 0].set(goals[:, 0] + 4.2)
        goals = goals.at[:, 1].set(goals[:, 1] + 3.0)
        #swap goals[0, :] and goals[2, :]
        temp = goals[0, :].copy()
        goals = goals.at[0, :].set(goals[2, :])
        goals = goals.at[2, :].set(temp)

        obs = jnp.ones((14, 2))
        obs_len = jnp.ones((14, 2))
        obs_theta = jnp.zeros((14, 1))

        # first four obstacles for room of size 6 x 6 centered at (0, 0)
        obs = obs.at[:4].set(jnp.array([[-3, 0], [0, 3], [3, 0], [0, -3]]))
        obs_len = obs_len.at[:4].set(jnp.array([[0.2, 6], [6, 0.2], [0.2, 6], [6, 0.2]]))

        # next obstacle at (-1.2, -3) of height 1.5
        obs_new = jnp.array([[-1.2, -2.25]]).reshape(2,)
        # obs = obs.at[4, :].set(obs_new)
        obs_rand = jr.uniform(use_key, (1,), minval=-0.1, maxval=0.1)
        obs_new = obs_new.at[0].set((obs_new[0] + obs_rand).reshape())
        obs = obs.at[4, :].set(obs_new)
        use_key, key = jr.split(key, 2)
        obs_theta_rand = jr.uniform(use_key, (1,), minval=-0.3, maxval=0.3)
        obs_theta = obs_theta.at[4].set(obs_theta_rand)
        use_key, key = jr.split(key, 2)
        obs_len_rand = jr.uniform(use_key, (1,), minval=-0.3, maxval=0.3) + 1.5
        obs_len = obs_len.at[4].set(jnp.array([[0.2, 1.5]]).reshape(2,))
        obs_len = obs_len.at[4, 1].set(obs_len_rand.reshape())
        obs = obs.at[4, 1].set(-3 + obs_len_rand.reshape() / 2)

        # next obstacle at (1.6, 0) of height 1.2
        use_key, key = jr.split(key, 2)
        obs_len_rand = jr.uniform(use_key, (1,), minval=-0.1, maxval=0.5) + 1.2
        obs = obs.at[5].set(jnp.array([[1.6, 0.6]]).reshape(2,))
        obs = obs.at[5, 1].set(obs_len_rand.reshape() / 2)
        use_key, key = jr.split(key, 2)
        obs_theta_rand = jr.uniform(use_key, (1,), minval=-0.4, maxval=0.4)
        obs_theta = obs_theta.at[5].set(obs_theta_rand)
        obs_len = obs_len.at[5].set(jnp.array([[0.2, 1.2]]).reshape(2,))
        obs_len = obs_len.at[5, 1].set(obs_len_rand.reshape())

        # next obstacle at (1.6, 0) of width 1.4
        # find bottom end point of the previous obstacle
        obs_end = obs[5, 0] + 0.5 * jnp.sin(obs_theta[5]) * obs_len[5, 1]
        obs = obs.at[6, 1].set(0)
        obs_len_x = (3.0 - obs_end)
        obs = obs.at[6, 0].set(obs_end.reshape() + (3-obs_end).reshape() / 2)
        obs_len = obs_len.at[6].set(jnp.array([[1.4, 0.2]]).reshape(2,))
        obs_len = obs_len.at[6, 0].set(obs_len_x.reshape())


        # next obstacle at (-2, 1.5) of height 1.5
        obs = obs.at[7].set(jnp.array([[-2, 2.25]]).reshape(2,))
        obs_len = obs_len.at[7].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        # next obstacle at (-0.5, 1.5) of height 1.5
        obs = obs.at[8].set(jnp.array([[-0.5, 2.25]]).reshape(2,))
        obs_len = obs_len.at[8].set(jnp.array([[0.2, 1.5]]).reshape(2,))

        # next obstacle at (-2, 1.5) of width 1.5
        obs = obs.at[9].set(jnp.array([[-1.25, 1.5]]).reshape(2,))
        obs_len = obs_len.at[9].set(jnp.array([[1.5, 0.2]]).reshape(2,))

        use_key, key = jr.split(key, 2)
        obs_x_rand = jr.uniform(use_key, (1,), minval=-0.2, maxval=0.5)

        obs = obs.at[7, 0].set(obs[7, 0] + obs_x_rand.reshape())
        obs = obs.at[8, 0].set(obs[8, 0] + obs_x_rand.reshape())
        obs = obs.at[9, 0].set(obs[9, 0] + obs_x_rand.reshape())

        # next 4 obstacles making a box of size 1 x 1 centered at (0, 0)
        obs = obs.at[10:].set(jnp.array([[-0.5, 0], [0, 0.5], [0.5, 0], [0, -0.5]]))
        obs_len = obs_len.at[10:].set(jnp.array([[0.2, 1], [1, 0.2], [0.2, 1], [1, 0.2]]))
        
        use_key, key = jr.split(key, 2)
        square_center_rand = jr.uniform(use_key, (2,), minval=-0.2, maxval=0.2)
        obs = obs.at[10:].set(obs[10:] + square_center_rand.reshape(1, 2))

        use_key, key = jr.split(key, 2)
        scale_factor = jr.uniform(use_key, (1,), minval=0.6, maxval=0.8)
        
        states = states * scale_factor
        obs = obs * scale_factor
        goals = goals * scale_factor
        obs_len = obs_len * 0.7
        states = states + 3 * scale_factor
        obs = obs + 3 * scale_factor
        goals = goals + 3 * scale_factor
        goals = goals.at[:, 0].set(goals[:, 0] + 0.3)

        # obstacles = self.create_obstacles(obs, obs_len[:, 0], obs_len[:, 1], obs_theta)
        # init_coll = True
        # while init_coll:
        #     use_key, key = jr.split(key, 2)
        #     state_pert = jr.uniform(use_key, (self.num_agents, 2), minval=-0.1, maxval=0.1)
        #     states_temp = states + state_pert
        #     inside_states = inside_obstacles(states_temp, obstacles, r=self._params["car_radius"] * 4)
        #     inter_agent_dist = jnp.linalg.norm(states_temp[:, None] - states_temp[None], axis=-1)
        #     inter_agent_safe = inter_agent_dist > 2 * self._params["car_radius"]
        #     if not jnp.any(inside_states) and jnp.all(inter_agent_safe):
        #         states = states_temp
        #         init_coll = False
        
        return states, goals, obs, obs_len, obs_theta

    def reset_corners(self):
        states =jnp.ones((self.num_agents, 2))

        states = states.at[:6, 0].set(jnp.array([12, 12.9, 13.8, 12, 12.9, 13.8]) - 4)
        states = states.at[:6, 1].set(jnp.array([12, 12, 12, 12.9, 12.9, 12.9]) - 4)

        states = states.at[6:12, 0].set(-1 * (jnp.array([12, 12.9, 13.8, 12, 12.9, 13.8]) - 4))
        states = states.at[6:12, 1].set(- 1 * (jnp.array([12, 12, 12, 12.9, 12.9, 12.9]) - 4))

        states = states.at[12:18, 0].set(jnp.array([-12, -12.9, -13.8, -12, -12.9, -13.8]) + 4)
        states = states.at[12:18, 1].set(jnp.array([12, 12, 12, 12.9, 12.9, 12.9]) - 4)

        states = states.at[18:24, 0].set(-1 * (jnp.array([-12, -12.9, -13.8, -12, -12.9, -13.8]) + 4))
        states = states.at[18:24, 1].set(- 1 * (jnp.array([12, 12, 12, 12.9, 12.9, 12.9]) - 4))

        goals = -1 * states

        obs = jnp.ones((16, 2))
        obs_len = jnp.ones((16, 2))
        obs_theta = jnp.zeros((16, 1))
        
        # first box centered at (-4.5, -4.5) of size 5 x 5
        obs = obs.at[:4].set(jnp.array([[-7, -4.5], [-4.5, -7], [-2, -4.5], [-4.5, -2]]))
        obs_len = obs_len.at[:4].set(jnp.array([[0.7, 5], [5, 0.7], [0.7, 5], [5, 0.7]]))

        # second box centered at (4.5, 4.5) of size 5 x 5
        obs = obs.at[4:8].set(jnp.array([[7, 4.5], [4.5, 7], [2, 4.5], [4.5, 2]]))
        obs_len = obs_len.at[4:8].set(jnp.array([[0.7, 5], [5, 0.7], [0.7, 5], [5, 0.7]]))

        # third box centered at (-4.5, 4.5) of size 5 x 5
        obs = obs.at[8:12].set(jnp.array([[-7, 4.5], [-4.5, 2], [-2, 4.5], [-4.5, 7]]))
        obs_len = obs_len.at[8:12].set(jnp.array([[0.7, 5], [5, 0.7], [0.7, 5], [5, 0.7]]))

        # fourth box centered at (4.5, -4.5) of size 5 x 5
        obs = obs.at[12:].set(jnp.array([[7, -4.5], [4.5, -7], [2, -4.5], [4.5, -2]]))
        obs_len = obs_len.at[12:].set(jnp.array([[0.7, 5], [5, 0.7], [0.7, 5], [5, 0.7]]))

        states = states * 0.2
        obs = obs * 0.2
        goals = goals * 0.2
        obs_len = obs_len * 0.2

        states = states + 3
        obs = obs + 3
        goals = goals + 3
        
        return states, goals, obs, obs_len, obs_theta
    
    def random_reset(self, key: Array):
        side_length = self.area_size
        
        use_key, key = jr.split(key, 2)

        states = jr.uniform(use_key, (self.num_agents, 2), minval=0, maxval=side_length)

        use_key, key = jr.split(key, 2)

        goals = jr.uniform(use_key, (self.num_agents, 2), minval=0, maxval=side_length)

        r = self._params["car_radius"]
        R = self._params["comm_radius"]

        # randomly generate positions of agents and goals
        max_iter = 1024  # maximum number of iterations to find a valid initial state/goal

        # randomly generate obstacles
        n_rng_obs = self._params["n_obs"]
        assert n_rng_obs >= 0
        obstacle_key, key = jr.split(key, 2)
        obs_pos = jr.uniform(obstacle_key, (n_rng_obs, 2), minval=0, maxval=side_length)
        length_key, key = jr.split(key, 2)
        obs_len = jr.uniform(
            length_key,
            (n_rng_obs, 2),
            minval=self._params["obs_len_range"][0],
            maxval=self._params["obs_len_range"][1],
        )
        theta_key, key = jr.split(key, 2)
        obs_theta = jr.uniform(theta_key, (n_rng_obs,1), minval=0, maxval=2 * np.pi)
        

        def get_node(reset_input: Tuple[int, int, Array, Array, Array]):  # key, node, all nodes
            i_iter, agent_id, this_key, _, all_nodes = reset_input
            use_key, this_key = jr.split(this_key, 2)
            i_iter += 1
            r_candidate = jr.uniform(use_key, (1,), minval=4*r, maxval=R)
            prev_agent_loc = all_nodes[jnp.where(agent_id >= 0, agent_id, 0), :2]
            use_key, this_key = jr.split(this_key, 2)
            theta_candidate = jr.uniform(use_key, (1,), minval=0, maxval=2 * np.pi)
            agent_candidate = prev_agent_loc + jnp.concatenate(
                [r_candidate * jnp.cos(theta_candidate), r_candidate * jnp.sin(theta_candidate)], axis=0
            )
            return i_iter, agent_id, this_key, agent_candidate, all_nodes

        def non_valid_node(reset_input: Tuple[int, int, Array, Array, Array]):  # key, node, all nodes
            i_iter, agent_id, _, node, all_nodes = reset_input
            dist_min = jnp.linalg.norm(all_nodes - node, axis=1).min()
            collide = dist_min <= self._params["car_radius"] * 4

            dist_connect = jnp.linalg.norm(node-all_nodes[jnp.where(agent_id >= 0, agent_id, 0), :2], axis=-1)
            disconnect = dist_connect > 0.9* self._params["comm_radius"]
            # inside = inside_obstacles(node, obstacles, r=self._params["car_radius"] * 4)
            outside = jnp.any(node < 0) | jnp.any(node > side_length)
            valid = ~(collide | outside | disconnect) | (i_iter >= max_iter)
            return ~valid

        def get_goal(reset_input: Tuple[int, int, Array, Array, Array, Array, Array]):
            # key, goal_candidate, agent_start_pos, all_goals
            i_iter, goal_id, this_key, _, agent, all_goals, _ = reset_input
            use_key, this_key = jr.split(this_key, 2)
            i_iter += 1
            prev_goal_loc = all_goals[jnp.where(goal_id>=0, goal_id, 0), :2]
            theta_candidate = jr.uniform(use_key, (1,), minval=0, maxval=2 * np.pi)
            use_key, this_key = jr.split(this_key, 2)
            r_candidate = jr.uniform(use_key, (1,), minval=4*r, maxval=R)
            goal_candidate = prev_goal_loc + jnp.concatenate(
                [r_candidate * jnp.cos(theta_candidate), r_candidate * jnp.sin(theta_candidate)], axis=0
            )

            if self.max_travel is None:
                return i_iter, goal_id, this_key, goal_candidate, agent, all_goals, all_goals
            else:
                return i_iter, goal_id, this_key, goal_candidate + agent, \
                       agent, all_goals, all_goals

        def non_valid_goal(reset_input: Tuple[int,int, Array, Array, Array, Array, Array]):
            # key, goal_candidate, agent_start_pos, all_goals
            i_iter, goal_id, _, goal, agent, all_goals, all_states = reset_input
            dist_min = jnp.linalg.norm(all_goals - goal, axis=1).min()
            collide = dist_min <= self._params["car_radius"] * 4
            # inside = inside_obstacles(goal, obstacles, r=self._params["car_radius"] * 4)
            outside = jnp.any(goal < 0) | jnp.any(goal > side_length)

            dist_connect = jnp.linalg.norm(goal-all_states[jnp.where(goal_id >= 0, goal_id, 0), :2], axis=-1)
            disconnect = dist_connect > 0.9 * self._params["comm_radius"]
            
            if self.max_travel is None:
                too_long = np.array(False, dtype=bool)
            else:
                too_long = jnp.linalg.norm(goal - agent) > self.max_travel
            
            valid = (~collide & ~outside & ~too_long & ~disconnect) | (i_iter >= max_iter)
            out = ~valid
            assert out.shape == tuple() and out.dtype == jnp.bool_
            return out

        def reset_body(reset_input: Tuple[int, Array, Array, Array]):
            # agent_id, key, states, goals
            agent_id, this_key, all_states, all_goals = reset_input
            agent_key, goal_key, this_key = jr.split(this_key, 3)
            agent_candidate = jr.uniform(agent_key, (2,), minval=0, maxval=side_length)
            # agent_candidate = jnp.zeros((2,))
            n_iter_agent, _, _, agent_candidate, _ = while_loop(
                cond_fun=non_valid_node, body_fun=get_node,
                init_val=(0, agent_id-1, agent_key, agent_candidate, all_states[:, :2])
            )
            all_states = all_states.at[agent_id, :2].set(agent_candidate)

            if self.max_travel is None:
                goal_candidate = jr.uniform(goal_key, (2,), minval=0, maxval=side_length)
            else:
                goal_candidate = jr.uniform(goal_key, (2,), minval=0, maxval=self.max_travel) + agent_candidate

            # goal_candidate = jnp.zeros((2,))

            n_iter_goal,_, _, goal_candidate, _, _, _ = while_loop(
                cond_fun=non_valid_goal, body_fun=get_goal,
                init_val=(0, agent_id-1, goal_key, goal_candidate, agent_candidate, all_goals[:, :2], all_states[:, :2])
            )

            all_goals = all_goals.at[agent_id, :2].set(goal_candidate)
            agent_id += 1

            # if no solution is found, start over
            agent_id = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * agent_id
            all_states = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_states
            all_goals = (1 - (n_iter_agent >= max_iter)) * (1 - (n_iter_goal >= max_iter)) * all_goals
            
            return agent_id, this_key, all_states, all_goals

        def reset_not_terminate(reset_input: Tuple[int, Array, Array, Array]):
            # agent_id, key, states, goals
            # jax.debug.print()
            agent_id, this_key, all_states, all_goals = reset_input

            return agent_id < self.num_agents

        _, _, states, goals = while_loop(
            cond_fun=reset_not_terminate, body_fun=reset_body, init_val=(0, key, states, goals))
        

        def get_obs(obs_reset_input: Tuple[int,  int, Array, Array, Array, Array, Array]):
            i_iter, obs_id, this_key, _, _, all_obs, _ = obs_reset_input
            use_key, this_key = jr.split(this_key, 2)
            i_iter += 1
            obs_candidate = jr.uniform(use_key, (2,), minval=0, maxval=side_length)
            length_key, this_key = jr.split(this_key, 2)
            obs_len = jr.uniform(
                length_key,
                (2,),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, this_key = jr.split(this_key, 2)
            obs_theta = jr.uniform(theta_key, (1,), minval=0, maxval=2 * np.pi)
            return i_iter, obs_id, this_key, obs_candidate, obs_len, all_obs, obs_theta

        def non_valid_obs(obs_reset_input: Tuple[int, int, Array, Array, Array, Array, Array]):
            i_iter, _, _, obs, obs_len, _, obs_theta = obs_reset_input
            obstacles = self.create_obstacles_single(obs, obs_len[0], obs_len[1], obs_theta)
            inside_states = inside_obstacles(states, obstacles, r=self._params["car_radius"] * 4)
            inside_states = inside_states.any()
            inside_goals = inside_obstacles(goals, obstacles, r=self._params["car_radius"] * 4)
            inside_goals = inside_goals.any()
            outside = jnp.any(obs < 0) | jnp.any(obs > side_length)
            valid = ~(inside_states | inside_goals | outside) | (i_iter >= max_iter)
            return ~valid
        
        def reset_obs_body(reset_input: Tuple[int, Array, Array, Array, Array]):
            obs_id, this_key, all_obs, all_obs_len, all_obs_theta = reset_input
            obs_key, this_key = jr.split(this_key, 2)
            obs_candidate = jr.uniform(obs_key, (2,), minval=0, maxval=side_length)
            length_key, this_key = jr.split(this_key, 2)
            obs_len = jr.uniform(
                length_key,
                (2,),
                minval=self._params["obs_len_range"][0],
                maxval=self._params["obs_len_range"][1],
            )
            theta_key, this_key = jr.split(this_key, 2)
            obs_theta = jr.uniform(theta_key, (1,), minval=0, maxval=2 * np.pi)
            # obs_candidate = obs_candidate.reshape((1, 2))
            n_iter_obs, obs_id, _, obs_candidate, obs_len, _, obs_theta = while_loop(
                cond_fun=non_valid_obs, body_fun=get_obs,
                init_val=(0, obs_id, obs_key, obs_candidate, obs_len, all_obs, obs_theta)
            )
            
            all_obs = all_obs.at[obs_id].set(obs_candidate)
            all_obs_len = all_obs_len.at[obs_id].set(obs_len)
            all_obs_theta = all_obs_theta.at[obs_id].set(obs_theta)

            obs_id += 1
            obs_id = (1 - (n_iter_obs >= max_iter)) * obs_id

            return obs_id, this_key, all_obs, all_obs_len, all_obs_theta
        
        def reset_obs_not_terminate(reset_input: Tuple[int, Array, Array, Array, Array]):
            obs_id, _, _, _, _ = reset_input
            return obs_id < n_rng_obs

        if n_rng_obs > 0:
            _, _, obs_pos, obs_len, obs_theta = while_loop(
                cond_fun=reset_obs_not_terminate, body_fun=reset_obs_body, init_val=(0, obstacle_key, obs_pos, obs_len, obs_theta))
        return states, goals, obs_pos, obs_len, obs_theta
    
    def reset(self, key: Array) -> GraphsTuple:
        self._t = 0
        if self.preset_reset:
            if self.preset_scene == 'box':
                print('resetting box')
                states, goals, obs_pos, obs_len, obs_theta = self.reset_box(key)
            elif self.preset_scene == 'original box':
                print('resetting original box')
                states, goals, obs_pos, obs_len, obs_theta = self.reset_box_original()
            elif self.preset_scene =='rand box':
                print('resetting box rand')
                states, goals, obs_pos, obs_len, obs_theta = self.reset_box_rand(key)
            elif self.preset_scene == 'corners':
                print('resetting corners')
                states, goals, obs_pos, obs_len, obs_theta = self.reset_corners()
            else:
                print("Preset scene not implemented, valid options are 'box', 'corners', 'double' \n")
                print('Using random reset')
                states, goals, obs_pos, obs_len, obs_theta = self.random_reset(key)
        else:
            print('random reset')
            states, goals, obs_pos, obs_len, obs_theta = self.random_reset(key)
        
        obstacles = self.create_obstacles(obs_pos, obs_len[:, 0], obs_len[:, 1], obs_theta)

        state_distances = jnp.linalg.norm(states[:, None, :2] - states[None, :, :2], axis=-1)
        state_distances = state_distances + jnp.eye(self.num_agents) * (self._params["comm_radius"] * 2 + 1)
        state_adjacency = state_distances < 1. * self._params["comm_radius"]
        env_states = self.EnvState(states, goals, obstacles)

        graph = self.get_graph(env_states, state_adjacency, goals)
        
        return graph

    def agent_step_euler(self, agent_states: AgentState, action: Action) -> AgentState:
        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)
        x_dot = action
        n_state_agent_new = x_dot * self.dt + agent_states
        assert n_state_agent_new.shape == (self.num_agents, self.state_dim)
        return self.clip_state(n_state_agent_new)

    def step(
            self, graph: EnvGraphsTuple, action: Action, get_eval_info: bool = False
    ) -> Tuple[EnvGraphsTuple, Reward, Cost, Done, Info]:
        self._t += 1

        # # calculate next graph
        # if self._prev_leader_mode and not self._leader_mode:
        #     agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        #     distances = jnp.linalg.norm(agent_states[:, None, :2] - agent_states[None, :, :2], axis=-1)
        #     distances = distances + jnp.eye(self.num_agents) * (self._params["comm_radius"] * 2 + 1)
        #     adjacency = distances < 1. * self._params["comm_radius"]
        #     # goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        #     goals = graph.xg
        # else:
        #     adjacency = graph.connectivity
        #     goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        adjacency = graph.connectivity
        goals = graph.type_states(type_idx=1, n_type=self.num_agents)
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)

        # the episode ends when reaching max_episode_steps
        done = jnp.array(False)

        # compute reward and cost
        reward = jnp.zeros(()).astype(jnp.float32)
        reward -= (jnp.linalg.norm(action - self.u_ref(graph), axis=1) ** 2).mean()
        cost = self.get_cost(graph)

        assert reward.shape == tuple()
        assert cost.shape == tuple()
        assert done.shape == tuple()

        next_state = self.EnvState(next_agent_states, goals, obstacles)

        info = {}
        if get_eval_info:
            # collision between agents and obstacles
            agent_pos = agent_states
            info["inside_obstacles"] = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])

        return self.get_graph(next_state, adjacency, graph.xg), reward, cost, done, info

    def get_cost(self, graph: GraphsTuple) -> Cost:
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        obstacles = graph.env_states.obstacle

        # collision between agents
        agent_pos = agent_states
        dist = jnp.linalg.norm(jnp.expand_dims(agent_pos, 1) - jnp.expand_dims(agent_pos, 0), axis=-1)
        dist += jnp.eye(self.num_agents) * 1e6
        collision = (self._params["car_radius"] * 2 > dist).any(axis=1)
        cost = collision.mean()

        # collision between agents and obstacles
        collision = inside_obstacles(agent_pos, obstacles, r=self._params["car_radius"])
        cost += collision.mean()

        return cost

    def render_video(
            self,
            rollout: RolloutResult,
            video_path: pathlib.Path,
            Ta_is_unsafe=None,
            viz_opts: dict = None,
            dpi: int = 100,
            **kwargs
    ) -> None:
        render_video(
            rollout=rollout,
            video_path=video_path,
            side_length=self.area_size,
            dim=2,
            n_agent=self.num_agents,
            n_rays=self.params["n_rays"],
            r=self.params["car_radius"],
            Ta_is_unsafe=Ta_is_unsafe,
            viz_opts=viz_opts,
            dpi=dpi,
            **kwargs
        )

    def edge_blocks(self, state: EnvState, lidar_data: Pos2d, connectivity) -> list[EdgeBlock]:
        n_hits = self._params["n_rays"] * self.num_agents

        # agent - agent connection
        agent_pos = state.agent
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        pos_diff = jnp.concatenate([pos_diff, connectivity[:, :, None], 1 - connectivity[:, :, None]], axis=-1)
        
        dist += jnp.eye(dist.shape[1]) * (self._params["comm_radius"] * 2 + 1)
        agent_agent_mask = jnp.less(dist, 2 * self._params["comm_radius"])
        id_agent = jnp.arange(self.num_agents)
        agent_agent_edges = EdgeBlock(pos_diff, agent_agent_mask, id_agent, id_agent)

        # agent - goal connection, clipped to avoid too long edges
        id_goal = jnp.arange(self.num_agents, self.num_agents * 2)
        agent_goal_mask = jnp.eye(self.num_agents)
        agent_goal_feats = state.agent[:, None, :] - state.goal[None, :, :]
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(agent_goal_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        agent_goal_feats = agent_goal_feats.at[:, :2].set(agent_goal_feats[:, :2] * coef)
        agent_goal_feats = jnp.concatenate([agent_goal_feats, jnp.zeros((self.num_agents, self.num_agents, 2))], axis=-1)
        agent_goal_edges = EdgeBlock(
            agent_goal_feats, agent_goal_mask, id_agent, id_goal
        )

        # agent - obs connection
        id_obs = jnp.arange(self.num_agents * 2, self.num_agents * 2 + n_hits)
        agent_obs_edges = []
        for i in range(self.num_agents):
            id_hits = jnp.arange(i * self._params["n_rays"], (i + 1) * self._params["n_rays"])
            lidar_feats = agent_pos[i, :] - lidar_data[id_hits, :]
            lidar_feats = jnp.concatenate([lidar_feats, jnp.zeros((self._params["n_rays"], 2))], axis=-1)
            lidar_dist = jnp.linalg.norm(lidar_feats, axis=-1)
            active_lidar = jnp.less(lidar_dist, self._params["comm_radius"] - 1e-1)
            agent_obs_mask = jnp.ones((1, self._params["n_rays"]))
            agent_obs_mask = jnp.logical_and(agent_obs_mask, active_lidar)
            agent_obs_edges.append(
                EdgeBlock(lidar_feats[None, :, :], agent_obs_mask, id_agent[i][None], id_obs[id_hits])
            )

        return [agent_agent_edges, agent_goal_edges] + agent_obs_edges

    def control_affine_dyn(self, state: State) -> [Array, Array]:
        assert state.ndim == 2
        f = jnp.zeros_like(state)
        g = jnp.eye(state.shape[1])
        g = jnp.expand_dims(g, axis=0).repeat(f.shape[0], axis=0)
        assert f.shape == state.shape
        assert g.shape == (state.shape[0], self.state_dim, self.action_dim)
        return f, g

    def add_edge_feats(self, graph: GraphsTuple, state: State) -> GraphsTuple:
        assert graph.is_single
        assert state.ndim == 2
        
        edge_feats = state[graph.receivers] - state[graph.senders]
        connectivity = graph.connectivity
        connectivity = connectivity[graph.receivers, graph.senders]
        edge_has_obs = graph.senders >= self.num_agents  # this edge has obstacles
        edge_feats = jnp.concatenate([edge_feats,  connectivity[:, None], 1 - connectivity[:, None]], axis=-1)
        edge_feats = jnp.where(~edge_has_obs[:, None], edge_feats, edge_feats.at[:, 2:].set(jnp.zeros((edge_feats.shape[0], 2))))
        feats_norm = jnp.sqrt(1e-6 + jnp.sum(edge_feats[:, :2] ** 2, axis=-1, keepdims=True))
        comm_radius = self._params["comm_radius"]
        safe_feats_norm = jnp.maximum(feats_norm, comm_radius)
        coef = jnp.where(feats_norm > comm_radius, comm_radius / safe_feats_norm, 1.0)
        edge_feats = edge_feats.at[:, :2].set(edge_feats[:, :2] * coef)
        
        return graph._replace(edges=edge_feats, states=state)

    def get_graph(self, state: EnvState, adjacency: Array = None, goals: Array = None) -> GraphsTuple:
        # node features
        n_hits = self._params["n_rays"] * self.num_agents
        n_nodes = 2 * self.num_agents + n_hits
        node_feats = jnp.zeros((self.num_agents * 2 + n_hits, 3))
        node_feats = node_feats.at[: self.num_agents, 2].set(1)  # agent feats
        node_feats = node_feats.at[self.num_agents: self.num_agents * 2, 1].set(1)  # goal feats
        node_feats = node_feats.at[-n_hits:, 0].set(1)  # obs feats

        # node type
        node_type = jnp.zeros(n_nodes, dtype=jnp.int32)
        node_type = node_type.at[self.num_agents: self.num_agents * 2].set(SingleIntegrator.GOAL)
        node_type = node_type.at[-n_hits:].set(SingleIntegrator.OBS)

        # edge blocks
        get_lidar_vmap = jax.vmap(
            ft.partial(
                get_lidar,
                obstacles=state.obstacle,
                num_beams=self._params["n_rays"],
                sense_range=self._params["comm_radius"],
            )
        )
        lidar_data = merge01(get_lidar_vmap(state.agent))
        edge_blocks = self.edge_blocks(state, lidar_data, connectivity=adjacency)

        # create graph
        return GetGraph(
            nodes=node_feats,
            node_type=node_type,
            edge_blocks=edge_blocks,
            env_states=state,
            states=jnp.concatenate([state.agent, state.goal, lidar_data], axis=0),
            connectivity=adjacency,
            xg=goals,
        ).to_padded()

    def state_lim(self, state: Optional[State] = None) -> Tuple[State, State]:
        lower_lim = jnp.ones(2) * -jnp.inf
        upper_lim = jnp.ones(2) * jnp.inf
        return lower_lim, upper_lim

    def action_lim(self) -> Tuple[Action, Action]:
        lower_lim = jnp.ones(2) * -1.0
        upper_lim = jnp.ones(2)
        return lower_lim, upper_lim

    def u_ref(self, graph: GraphsTuple) -> Action:
        agent = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal = graph.type_states(type_idx=1, n_type=self.num_agents)
        error = goal - agent
        error_max = jnp.abs(error / jnp.linalg.norm(error, axis=-1, keepdims=True) * self._params["comm_radius"])
        error = jnp.clip(error, -error_max, error_max)
        return self.clip_action(error @ self._K.T)

    def forward_graph(self, graph: GraphsTuple, action: Action) -> GraphsTuple:
        # calculate next graph
        agent_states = graph.type_states(type_idx=0, n_type=self.num_agents)
        goal_states = graph.type_states(type_idx=1, n_type=self.num_agents)
        obs_states = graph.type_states(type_idx=2, n_type=self._params["n_rays"] * self.num_agents)
        action = self.clip_action(action)

        assert action.shape == (self.num_agents, self.action_dim)
        assert agent_states.shape == (self.num_agents, self.state_dim)

        next_agent_states = self.agent_step_euler(agent_states, action)
        next_states = jnp.concatenate([next_agent_states, goal_states, obs_states], axis=0)

        next_graph = self.add_edge_feats(graph, next_states)

        return next_graph

    @ft.partial(jax.jit, static_argnums=(0,))
    def safe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)

        # agents are not colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        safe_agent = jnp.greater(dist, self._params["car_radius"] * 2.5)
        safe_agent = jnp.min(safe_agent, axis=1)

        safe_obs = jnp.logical_not(
            inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"] * 1.5)
        )

        safe_mask = jnp.logical_and(safe_agent, safe_obs)

        return safe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def disconnect_mask(self, graph: GraphsTuple) -> Array:
        agent_state = graph.type_states(type_idx=0, n_type=self.num_agents)
        agent_pos = agent_state[:, :2]
        adjacency = graph.connectivity
        # agents are colliding
        agent_pos_diff = agent_pos[None, :, :] - agent_pos[:, None, :]
        agent_dist = jnp.linalg.norm(agent_pos_diff, axis=-1)
        
        disconnect_agent = jnp.greater(agent_dist, self._params["comm_radius"])
        
        disconnect_agent_mask = jnp.logical_and(disconnect_agent, adjacency)
        disconnect_agent_mask = jnp.max(disconnect_agent_mask, axis=1)

        # breakpoint_if(disconnect_agent_mask.any())

        return disconnect_agent_mask  # | unsafe_stop
    
    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        unsafe_mask = jnp.logical_or(unsafe_agent, unsafe_obs)
        disconnect_mask = self.disconnect_mask(graph)
        unsafe_mask = jnp.logical_or(unsafe_mask, disconnect_mask * self.use_connect)

        return unsafe_mask

    @ft.partial(jax.jit, static_argnums=(0,))
    def unsafe_test_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)

        # agents are colliding
        pos_diff = agent_pos[:, None, :] - agent_pos[None, :, :]  # [i, j]: i -> j
        dist = jnp.linalg.norm(pos_diff, axis=-1)
        dist = dist + jnp.eye(dist.shape[1]) * (self._params["car_radius"] * 2 + 1)  # remove self connection
        unsafe_agent = jnp.less(dist, self._params["car_radius"] * 2)
        unsafe_agent = jnp.max(unsafe_agent, axis=1)

        # agents are colliding with obstacles
        unsafe_obs = inside_obstacles(agent_pos, graph.env_states.obstacle, self._params["car_radius"])

        unsafe_mask = jnp.logical_or(unsafe_agent, unsafe_obs)

        return unsafe_mask
    
    def collision_mask(self, graph: GraphsTuple) -> Array:
        return self.unsafe_mask(graph)

    def finish_mask(self, graph: GraphsTuple) -> Array:
        agent_pos = graph.type_states(type_idx=0, n_type=self.num_agents)[:, :2]
        goal_pos = graph.xg
        reach = jnp.linalg.norm(agent_pos - goal_pos, axis=1) < self._params["car_radius"] * 2
        return reach

import pickle
import argparse
import os
from typing import NamedTuple
import numpy as np
import ipdb
import jax
from gcbfplus.env.utils import TrajLog
from gcbfplus.utils.graph import GraphsTuple
from gcbfplus.env.obstacle import Obstacle, Rectangle
from gcbfplus.utils.typing import Action, Array, Cost, Done, Info, Pos2d, Reward, State, AgentState

class EnvState(NamedTuple):
    agent: State
    goal: State
    obstacle: Obstacle
#
# class Test(NamedTuple):
#     a: int
#     b: int
#
#
# test = Test(1, 2)
# pickle.dump(test, open('test.pkl', 'wb'))
#
# del test
#
# test = pickle.load(open('test.pkl', 'rb'))
# print(test)


def load_traj(traj_path):
    traj: TrajLog = pickle.load(open(traj_path, 'rb'))
    graphs = GraphsTuple(*traj.graphs)
    return graphs, traj.leader, traj.leader_dir

def main(args):
    
    graphs, leader, leader_dir = load_traj(os.path.join(args.path, f'traj_log_{args.traj_id}.pkl'))
    # graph_flat = graphs.tree_flatten_with_keys()
    # print(graph_flat)
    # print(type(graph_flat))
    graph_flat, aux_data = graphs.tree_flatten_with_keys()
    # print(graph_flat)
    # print(aux_data)
    # print(graph_flat[0])
    n_node, n_edge, nodes, edges, states, receivers, senders, node_type, env_states, connectivity, xg = graph_flat
    
    _, data_n_node = n_node
    _, data_n_edge = n_edge
    _, data_nodes = nodes
    _, data_edges = edges
    _, data_states = states
    _, data_receivers = receivers
    _, data_senders = senders
    _, data_node_type = node_type
    _, data_env_states = env_states
    agent_state, goal_state, obstacles = data_env_states
    _, data_connectivity = connectivity
    _, data_xg = xg

    rollout_len = data_states.shape[0]
    agent_state = np.array(agent_state)
    goal_state = np.array(goal_state)
    
    num_agents = agent_state.shape[1]
    lidar_data = data_states[:, num_agents * 2:, :]
    print('lidar data shape: ', lidar_data.shape)
    print(lidar_data)
    connectivity = np.array(data_connectivity)
    actual_goal = np.array(data_xg)
    temp_goal = np.array(goal_state)
    leader = np.array(leader)
    leader_dir = np.array(leader_dir)
    edges = np.array(data_edges)
    obs_edges = edges[2*num_agents:, :2]
    receivers = np.array(data_receivers)
    senders = np.array(data_senders)
    n_rays = 32
    n_hits = num_agents * n_rays
    dim=2
    all_pos = data_states[:, :num_agents * 2 + n_hits, :dim]
    print('sender shape: ', senders.shape)
    print('receiver shape: ', receivers.shape)
    edge_index = np.stack([senders[:, :, None], receivers[:, :, None]], axis=-1)
    print('edge_index shape: ', edge_index.shape)
    edge_index = edge_index.squeeze(-2)
    e_edge_index = edge_index.copy()
    print('all pos shape: ', all_pos.shape)
    e_start, e_end = all_pos[e_edge_index[:, :, 0]], all_pos[e_edge_index[:, :, 1]]
    print('e_start shape: ', e_start.shape)
    print('e_end shape: ', e_end.shape)

    e_lines = np.stack([e_start, e_end], axis=-1)
    e_is_obs = senders < 2 * num_agents + n_hits
    # e_is_obs = np.logical_and(e_is_obs, senders >= 2 * num_agents)
    # e_is_obs = e_is_obs[~is_pad]
    obs_edges = e_lines[e_is_obs]
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True)
    parser.add_argument('--traj-id', type=int, default=0)
    args = parser.parse_args()
    # with ipdb.slaunch_ipdb_on_exception():
    main(args)
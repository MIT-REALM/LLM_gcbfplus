import jax.numpy as jnp
import jax.tree_util as jtu
import jax
import numpy as np
import socket
import matplotlib.pyplot as plt
import functools as ft
import seaborn as sns
import optax

from typing import Callable, TYPE_CHECKING
from matplotlib.colors import CenteredNorm

from ..utils.typing import PRNGKey
from ..utils.graph import GraphsTuple
from ..utils.utils import tree_index
from .data import Rollout


if TYPE_CHECKING:
    from ..env import MultiAgentEnv
else:
    MultiAgentEnv = None

def multi_control(
        env: MultiAgentEnv,
        actor: Callable,
        key: PRNGKey,
        graph: GraphsTuple,
):
    """
    Get all possible actions and corresponding rewards.

    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, PRNGKey] -> [Action, LogPi]
    key: PRNGKey

    Returns
    -------
    graph with max reward after reconfiguration
    """

    all_graphs, _, eig_Ls = env.get_all_graphs(graph)
    get_actions = jax.vmap(actor, in_axes=(0, None))
    # get_actions = jax.vmap(get_actions, in_axes=(0, None))
    actions, _ = get_actions(all_graphs, key)
    get_data = jax.vmap(env.step_multi, in_axes=(None, 0))
    rewards = get_data(graph, actions)
    eig_Ls = jnp.asarray(eig_Ls)
    rewards = jnp.asarray(rewards)
    
    scaled_rewards = jnp.where(eig_Ls > 0.1, rewards * eig_Ls, -1e1)
    max_reward_ind = jnp.argmax(scaled_rewards)

    max_reward_graph = tree_index(all_graphs, max_reward_ind)
    # max_reward_action = tree_index(actions, max_reward_ind)
    # max_reward_log_pi = tree_index(log_pis, max_reward_ind)
    return max_reward_graph

# def leader_control(
#         env: MultiAgentEnv,
#         actor: Callable,
#         key: PRNGKey,
#         graph: GraphsTuple,
# ):
#     """
#     Check if the average speed is below a threshold.

#     Parameters
#     ----------
#     env: MultiAgentEnv
#     actor: Callable, [GraphsTuple, PRNGKey] -> [Action, LogPi]
#     key: PRNGKey

#     Returns
#     -------
#     graph with leader-follower assignment
#     """
#     actions, _ = actor(graph, key)
#     action_norm = jnp.linalg.norm(actions.nodes, axis=-1)
#     avg_speed = jnp.mean(action_norm)
#     goal_pos = graph.env_states.goal[:, :2]
#     agent_pos = graph.type_states(type_idx=0, n_type=env.num_agents)[:, :2]
#     closest_agent = jnp.argmin(jnp.linalg.norm(goal_pos - agent_pos, axis=-1))
#     leader = closest_agent
#     graph = jnp.where(avg_speed < 0.1, env.leader_follower_assign(graph, leader), graph)

#     return graph

def rollout(
        env: MultiAgentEnv,
        actor: Callable,
        key: PRNGKey
) -> Rollout:
    """
    Get a rollout from the environment using the actor.

    Parameters
    ----------
    env: MultiAgentEnv
    actor: Callable, [GraphsTuple, PRNGKey] -> [Action, LogPi]
    key: PRNGKey

    Returns
    -------
    data: Rollout
    """
    key_x0, key = jax.random.split(key)
    init_graph = env.reset(key_x0)

    def body(graph, key_):
        # if env._use_leader:
        #     key, key_ = jax.random.split(key_)
        #     graph = leader_control(env, actor, key, graph)
            
        if env._reconfig_connect: # and not env._use_leader:
            key, key_ = jax.random.split(key_)
            graph = multi_control(env, actor, key, graph)

        action, log_pi = actor(graph, key_)
        next_graph, reward, cost, done, info = env.step(graph, action)
        return next_graph, (graph, action, reward, cost, done, log_pi, next_graph)

    keys = jax.random.split(key, env.max_episode_steps)
    final_graph, (graphs, actions, rewards, costs, dones, log_pis, next_graphs) = \
        jax.lax.scan(body, init_graph, keys, length=env.max_episode_steps)
    data = Rollout(graphs, actions, rewards, costs, dones, log_pis, next_graphs)
    return data


def has_nan(x):
    return jtu.tree_map(lambda y: jnp.isnan(y).any(), x)


def has_any_nan(x):
    return jnp.array(jtu.tree_flatten(has_nan(x))[0]).any()


def compute_norm(grad):
    return jnp.sqrt(sum(jnp.sum(jnp.square(x)) for x in jtu.tree_leaves(grad)))


def compute_norm_and_clip(grad, max_norm: float):
    g_norm = compute_norm(grad)
    clipped_g_norm = jnp.maximum(max_norm, g_norm)
    clipped_grad = jtu.tree_map(lambda t: (t / clipped_g_norm) * max_norm, grad)

    return clipped_grad, g_norm


def tree_copy(tree):
    return jtu.tree_map(lambda x: x.copy(), tree)


def empty_grad_tx() -> optax.GradientTransformation:
    def init_fn(params):
        return optax.EmptyState()

    def update_fn(updates, state, params=None):
        return None, None

    return optax.GradientTransformation(init_fn, update_fn)


def jax2np(x):
    return jtu.tree_map(lambda y: np.array(y), x)


def np2jax(x):
    return jtu.tree_map(lambda y: jnp.array(y), x)


def is_connected():
    try:
        sock = socket.create_connection(("www.google.com", 80))
        if sock is not None:
            sock.close()
        return True
    except OSError:
        pass
    print('No internet connection')
    return False


def plot_cbf(
        fig: plt.Figure,
        cbf: Callable,
        env: MultiAgentEnv,
        graph: GraphsTuple,
        agent_id: int,
        x_dim: int,
        y_dim: int,
) -> plt.Figure:
    ax = fig.gca()
    n_mesh = 30
    low_lim, high_lim = env.state_lim(graph.states)
    x, y = jnp.meshgrid(
        jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh),
        jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    )
    states = graph.states

    # generate new states
    plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    plot_states = plot_states.at[:, :, agent_id, x_dim].set(x)
    plot_states = plot_states.at[:, :, agent_id, y_dim].set(y)

    get_new_graph = env.add_edge_feats
    get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
    new_graph = get_new_graph_vmap(plot_states)
    h = jax.vmap(jax.vmap(cbf))(new_graph)[:, :, agent_id, :].squeeze(-1)
    plt.contourf(x, y, h, cmap=sns.color_palette("rocket", as_cmap=True), levels=15, alpha=0.5)
    plt.colorbar()
    plt.contour(x, y, h, levels=[0.0], colors='blue')
    ax.set_xlim(low_lim[0], high_lim[0])
    ax.set_ylim(low_lim[1], high_lim[1])
    plt.axis('off')

    return fig


def get_bb_cbf(cbf: Callable, env: MultiAgentEnv, graph: GraphsTuple, agent_id: int, x_dim: int, y_dim: int):
    n_mesh = 20
    low_lim = jnp.array([0, 0])
    high_lim = jnp.array([env.area_size, env.area_size])
    b_xs = jnp.linspace(low_lim[x_dim], high_lim[x_dim], n_mesh)
    b_ys = jnp.linspace(low_lim[y_dim], high_lim[y_dim], n_mesh)
    bb_Xs, bb_Ys = jnp.meshgrid(b_xs, b_ys)
    states = graph.states

    # generate new states
    bb_plot_states = states[None, None, :, :].repeat(n_mesh, axis=0).repeat(n_mesh, axis=1)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, x_dim].set(bb_Xs)
    bb_plot_states = bb_plot_states.at[:, :, agent_id, y_dim].set(bb_Ys)

    get_new_graph = env.add_edge_feats
    get_new_graph_vmap = jax.vmap(jax.vmap(ft.partial(get_new_graph, graph)))
    bb_new_graph = get_new_graph_vmap(bb_plot_states)
    bb_h = jax.vmap(jax.vmap(cbf))(bb_new_graph)[:, :, agent_id, :].squeeze(-1)
    assert bb_h.shape == (n_mesh, n_mesh)
    return b_xs, b_ys, bb_h


def centered_norm(vmin: float | list[float], vmax: float | list[float]):
    if isinstance(vmin, list):
        vmin = min(vmin)
    if isinstance(vmax, list):
        vmin = max(vmax)
    halfrange = max(abs(vmin), abs(vmax))
    return CenteredNorm(0, halfrange)

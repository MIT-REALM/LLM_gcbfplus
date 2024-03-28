import datetime
import functools as ft
import os
import jax
import jax.numpy as jnp
import numpy as np

from gcbfplus.env.utils import get_lidar

from ..utils.graph import GraphsTuple

from openai import OpenAI


def get_response(prompt, model, LLM_calls=1):
    """
    Get the response from the LLM model given the prompt.
    Args:
        prompt: the prompt to be sent to the LLM model
        model: the model to be used for the LLM
        LLM_calls: the number of calls to the LLM model
        fixed_prompts: the fixed prompts to be used for the LLM model (problem description and in-context examples)
        new_prompt: the new prompt to be used for the LLM model (current deadlock environment description)
    Returns:
        return_message: JSON output with leader assignment from the LLM model
        response: the complete response object from the LLM model
    """
    client = OpenAI(
        # This is the default and can be omitted
        api_key=os.getenv("OPENAI_API_KEY"),
    )
    if LLM_calls == 1:
        tmp = 0.0
    else:
        if 'gpt-4' in model:
            tmp = 0.7
        else:
            tmp = 1.0
    print('prompting LLM with model: ', model, ' and temperature: ', tmp)
    if tmp == 0.0:
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=tmp,
            seed=100,
            response_format={"type": "json_object"},
            max_tokens=100,
            # logprobs=True,
        )
    else:    
        response = client.chat.completions.create(
            model=model,
            messages=prompt,
            temperature=tmp,
            # seed=100,
            response_format={"type": "json_object"},
            max_tokens=100,
            # logprobs=True,
        )
    print('response received: ', response)
    return_message = response.choices[0].message.content
    return return_message, response 


def get_info_single_graph(graph, n_agent):
    """
    Get the information of a single graph.
    Args:
        graph: the graph to get the information from
        n_agent: the number of agents in the graph
    Returns:
        agent_states: the states of the agents in the graph
        global_goal_states: the global goal states of the agents in the graph
        temp_goal_states: the temporary goal states of the agents in the graph
        connectivity: the connectivity of the agents in the graph
        lidar_data: the lidar data of the agents in the graph
    """
    agent_states = graph.type_states(0, n_agent)
    global_goal_states = graph.xg
    temp_goal_states = graph.type_states(1, n_agent)
    connectivity = graph.connectivity
    obstacles = graph.env_states.obstacle
    get_lidar_vmap = jax.vmap(
        jax.jit(ft.partial(
            get_lidar,
            obstacles=obstacles,
            num_beams=32,
            sense_range=0.5,
        ))
    )
    lidar_data = get_lidar_vmap(agent_states)  # (n_agent, n_rays, 2)
    return agent_states, global_goal_states, temp_goal_states, connectivity, lidar_data

def create_user_prompt_fn(graph, n_agent, iter):
    """
    Create the user prompt for the LLM model.
    Args:
        graph: the graph to create the user prompt from
        n_agent: the number of agents in the graph
        iter: the iteration number
        use_local_leader: whether to use a local leader
        deadlock_graphs: the deadlock graphs
        use_normalized_data: whether to use normalized data
    Returns:
        message: the user prompt message
        far_agent_indices: the indices of the far agents
    """
    # Example prompt 
    #Number of agents***5***Safety radius***0.1***Connectivity radius***1.7
    #***Agent***Id: 1, current state: (-2.28,-2.17,0.60), goal location: (1.70,1.30), obstacle seen at: (-3.00,-2.16), Id: 2, current state: (-1.99,-1.99,1.88), goal location: (2.50,0.50), obstacle seen at: (-1.20,-2.04), Id: 3, current state: (-2.21,-1.46,0.88), goal location: (1.70,0.50), obstacle seen at: (-3.00,-1.50), Id: 4, current state: (-1.82,-1.02,1.26), goal location: (2.50,1.30), obstacle seen at: (-1.20,-1.02), Id: 5, current state: (-1.59,-0.48,0.53), goal location: (2.10,2.00), obstacle seen at: (-1.20,-0.48), ***Connections***(0,1),(1,2),(2,3),(3,4),
    num_agents = n_agent
    agent_states, global_goal_states, _, connectivity, lidar_data = get_info_single_graph(graph, n_agent)
    x = agent_states[:, 0]
    y = agent_states[:, 1]
    obs = lidar_data
    
    xg = global_goal_states
    E_time = connectivity
    new_prompt = '***Name***Env:' + str(iter)
    new_prompt += '***Number of agents***' + str(n_agent)
    new_prompt +='***Safety radius***0.05***Connectivity radius***0.5***'
    new_prompt += 'Agents'
    for i in range(n_agent):
        new_prompt += '***AgentId***' + str(i + 1) + '***current state***(' + '{:.2f}'.format(x[i].item()) + ',' + '{:.2f}'.format(y[i].item()) + ')***goal location***(' + '{:.2f}'.format(xg[i][0].item()) + ',' + '{:.2f}'.format(xg[i][0].item()) + ')' 
        # obs_known = jax.numpy.linalg.norm(obs[i, :, :] - jnp.array([x[i], y[i]]), axis=-1) < 0.5
        obs_known_dist = jax.numpy.linalg.norm(obs[i, :, :] - jnp.array([x[i], y[i]]), axis=-1)
        obs_known_order = jax.numpy.argsort(obs_known_dist)
        # select obs that are at most 1 away 
        num_obs_seen = jax.numpy.sum((obs_known_dist < 1).astype(jax.numpy.int32))
        num_obs_seen = min(num_obs_seen, 3)
        new_prompt += '***obstacles seen at***['
        if num_obs_seen == 0:
            new_prompt += 'None'
        else:
            for l in range(num_obs_seen):
                ind = obs_known_order[l]
                if l < num_obs_seen - 1:
                    new_prompt += '(' + '{:.2f}'.format(obs[i, ind, 0].item()) + ',' + '{:.2f}'.format(obs[i, ind, 1].item()) + '),'
                else:
                    new_prompt += '(' + '{:.2f}'.format(obs[i, ind, 0].item()) + ',' + '{:.2f}'.format(obs[i, ind, 1].item()) + ')'
        new_prompt += ']'
    # new_prompt += '***Connections***['
    # adjacency_pairs = []
    # for j in range(num_agents):
    #     for l in range(j+1, num_agents):
    #         if E_time[j, l] > 0:
    #             adjacency_pairs.append((j + 1, l + 1))
    # for j in range(len(adjacency_pairs)):
    #     if j < len(adjacency_pairs) - 1:
    #         new_prompt += '(' + str(adjacency_pairs[j][0]) + ',' + str(adjacency_pairs[j][1]) + '),'
    #     else:
    #         new_prompt += '(' + str(adjacency_pairs[j][0]) + ',' + str(adjacency_pairs[j][1]) + ')'
    #     # new_prompt += '"'
    #     # new_prompt += '}'
    # new_prompt += ']'
    message = {"role": "user", "content": new_prompt}

    return message

def create_assistant_prompt_fn(leader_id, dir, graph, n_agent):
    """
    Create the assistant prompt for the LLM model.
    Args:
        leader_id: the id of the leader
        dir: the direction of the leader
        graph: the graph to create the assistant prompt from
        n_agent: the number of agents in the graph
    Returns:
        output_message: the assistant prompt message
    """
    # Example prompt: {"Leader": 1, "Direction": "To right"}
    
    leader_i = leader_id
    leader_dir_i = dir
    #    str_output = '"Output": {'
    lead_rs = leader_i
    str_output = ' {"Leader": ' + str(lead_rs.item() + 1) + ','    
    if graph is None:
        if jnp.linalg.norm(leader_dir_i - jnp.array([1, 0])) < 0.1:
            str_output += '"Direction": "To right"'
        elif jnp.linalg.norm(leader_dir_i - jnp.array([-1, 0])) < 0.1:
            str_output += '"Direction": "To left"'
        else:
            str_output += '"Direction": "To goal"'
    else:
        actual_goal = graph.xg[leader_id]
        state_leader = graph.type_states(0, n_agent)
        state_leader = state_leader[leader_id]
        temp_goal = graph.type_states(1, n_agent)
        temp_goal = temp_goal[leader_id]
        if jnp.linalg.norm(temp_goal - actual_goal) < 0.1:
            str_output += '"Direction": "To goal"'
        else:
            dir = temp_goal - state_leader
            dir_ac = actual_goal - state_leader
            if jnp.cross(dir, dir_ac) > 0:
                str_output += '"Direction": "To right"'
            else:
                str_output += '"Direction": "To left"'
    str_output += '}'
    output_message = {"role": "assistant", "content": str_output}
    return output_message

def nominal_leader_dir_fn():
    return jnp.array(0), jnp.array([1, 0])

def LLM_leader_dir_fn(response, graph, n_agent):
    """
    Get the leader id and direction from the LLM response.
    Args:
        response: the response from the LLM model
        graph: the graph to get the leader id and direction from
        n_agent: the number of agents in the graph
    Returns:
        leader_id: the id of the leader
        LLM_direction: the direction of the leader
    Returns None, None if there is an error in the LLM response.
    """
    try:
        LLM_response = response
        LLM_response = LLM_response.replace('"', "'")
        LLM_response = LLM_response.replace('\n', '')
        LLM_response = LLM_response.replace('\n', '')
        LLM_response = LLM_response.replace(' ', '')
        LLM_response = LLM_response.replace('"', '')
        # find multi-digit integer values in the response and assign as leader
        leader = [int(s) for s in LLM_response if s.isdigit()]
        len_leader = len(leader)
        leader_id = 0
        agent_states = graph.type_states(0, n_agent)
        for i in range(len_leader):
            leader_id += 10 ** (len_leader - i - 1) * leader[i]
        leader_id = int(leader_id) - 1
        if 'left' in LLM_response:
            LLM_direction = graph.xg[leader_id] - agent_states[leader_id]
            # rotate 90 degrees
            LLM_direction = jnp.array([-LLM_direction[1], LLM_direction[0]])
        elif 'right' in LLM_response:
            LLM_direction = graph.xg[leader_id] - agent_states[leader_id]
            # rotate -90 degrees
            LLM_direction = jnp.array([LLM_direction[1], -LLM_direction[0]])
        elif 'goal' in LLM_response:
            LLM_direction = graph.xg[leader_id] - agent_states[leader_id]
            # LLM_direction = jnp.array([0, 1])
        else:
            LLM_direction = graph.xg[leader_id] - agent_states[leader_id]
            # LLM_direction = jnp.array([1, 0])
        LLM_direction = LLM_direction / (jnp.linalg.norm(LLM_direction) + 1e-6)
    except:
        print('Error in LLM_leader_dir_fn')
        return None, None
    return jnp.array(leader_id), LLM_direction

def create_init_prompt(num_incontext, preset=False):
    """
    Create the initial prompt for the LLM model.
    Args:
        num_incontext: the number of in-context examples
        preset: whether to use a preset prompt
    Returns:
        log_message: the log message
        log_dir: the log directory
    """
    # start_time = datetime.datetime.now()
    # start_time = start_time.strftime("%Y%m%d%H%M%S")

    file_name = 'LLM_traj_tests'
    if preset:
        log_dir = f"{'LLM_files'}/num_incontext_{num_incontext}/{file_name}"
    else:
        log_dir = f"{'LLM_files'}/random_num_incontext_{num_incontext}/{file_name}"
    os.makedirs(log_dir, exist_ok=True)
    
    prompt_file = 'LLM_files/LLM_in_context_prompts.txt'

    with open(prompt_file, "r") as f:
        prompt = f.read()

    prompts = prompt.split("\n")

    log_message = []
    system_message = {"role": "system", "content": "You are a helpful assistant for multi-robot deadlock resolution situations designed to output a JSON output."}
    log_message.append(system_message)
    for p in prompts:
        log_message.append({"role": "user", "content": p})
        
    return log_message, log_dir

def barebone_message(LLM_response):
    """
    Clean the LLM response.
    """
    LLM_response = LLM_response.replace('"', "'")
    LLM_response = LLM_response.replace('\n', '')
    LLM_response = LLM_response.replace('\n', '')
    LLM_response = LLM_response.replace(' ', '')
    LLM_response = LLM_response.replace('"', '')
    return LLM_response

def find_median(messages, graph, n_agent):
    """
    Find the median message from the LLM responses.
    Args:
        messages: the messages from the LLM responses
        graph: the graph to get the leader id and direction from
        n_agent: the number of agents in the graph
    Returns:
        leader_id: the id of the leader from the most frequent message
        LLM_direction: the direction of the leader from the most frequent message
    """
    length = len(messages)
    mess_append = []
    leader_id = None
    LLM_direction = None
    for i in range(length):
        response = barebone_message(messages[i])
        mess_append.append(response)
    
    sim_count = jnp.zeros((length, 1))
    for i in range(length):
        sim_count = sim_count.at[i].set(mess_append.count(mess_append[i]))
        if any(sim_count > length / 2):
            break
    sim_count_max = jnp.argmax(sim_count)
    median_message = mess_append[int(sim_count_max)] 

    leader_id, LLM_direction = LLM_leader_dir_fn(median_message, graph, n_agent)
    # print(asas)

    # print('median leader_id: ', leader_id, ' median direction: ', LLM_direction)
    # print(asas)
    leader_id = leader_id.astype(jnp.int32)
    return leader_id, LLM_direction

def get_leader_id_dir(graph: GraphsTuple):
    """
    Get the id of the leader of the graph, return -1 if no leader is found.
    Also return the moving direction of the leader (only meaningful when leader_id is not -1).
    """
    true_goals = graph.xg
    cur_goals = graph.type_states(type_idx=1, n_type=true_goals.shape[0])
    agent_states = graph.type_states(type_idx=0, n_type=true_goals.shape[0])
    leader_id = jnp.argmax(jnp.linalg.norm(cur_goals - agent_states, axis=-1))
    leader_id = jnp.where(jnp.allclose(cur_goals, true_goals), jnp.array([-1]), leader_id)
    leader_dir = cur_goals[leader_id] - agent_states[leader_id]
    return leader_id, leader_dir
    
def leader_graph(graph, num_agents, jit_policy, keep_mode, jit_leader_follower_assign, leader_model, kk ,log_dir, num_runtime_incontext_prompts, leader_control, policy,prompt,all_prompts,LLM_calls,reset_graph, leader_assign_count, t_mode, prompt_time_gap, num_LLM_calls, sent_token_count, received_token_count):
    """
    Assign the leader to the agents in the graph.
    Args:
        graph: the graph to assign the leader to
        num_agents: the number of agents in the graph
        jit_policy: the control policy function
        keep_mode: the number of steps to keep the leader
        jit_leader_follower_assign: the leader follower assignment function
        leader_model: the leader model to be used
        kk: the current iteration number
        log_dir: the log directory
        num_runtime_incontext_prompts: the number of runtime in-context prompts
        leader_control: the leader control function
        policy: the policy to be used for the leader control
        prompt: the prompt to be used for the LLM model
        all_prompts: all the prompts for bookkeeping
        LLM_calls: the number of LLM calls
        reset_graph: the function to reset the graph
        leader_assign_count: the number of leader assignments
        t_mode: the mode of the graph
        prompt_time_gap: the time gap between prompts
        num_LLM_calls: the number of LLM calls
        sent_token_count: the number of tokens sent to the LLM model
        received_token_count: the number of tokens received from the LLM model
    Returns:
        graph: the graph with the leader assigned
        num_LLM_calls: the number of LLM calls
        prompt: the prompt to be used for the LLM model
        all_prompts: all the prompts for bookkeeping
        leader_assign_count: the number of leader assignments
        t_mode: the mode of the graph
        prompt_time_gap: the time gap between prompts
        sent_token_count: the number of tokens sent to the LLM model
        received_token_count: the number of tokens received from the LLM model
    """
    goals = graph.type_states(type_idx=1, n_type=num_agents)
    agent_states = graph.type_states(type_idx=0, n_type=num_agents)
    
    actions = jit_policy(graph)
    action_norm = jnp.linalg.norm(actions, axis=-1)
    avg_speed = jnp.mean(action_norm)
    # mean_speed.append(avg_speed)
    low_speed = avg_speed < 0.2
    far_from_goal = jnp.linalg.norm(graph.xg - agent_states[:, :2], axis=-1).mean() > 0.5

    use_leader_mode = low_speed & far_from_goal

    # judge if the current graph is in the leader mode
    in_leader_mode = ~jnp.allclose(goals, graph.xg)

    if in_leader_mode and prompt_time_gap < keep_mode:
            # keep the same leader
            cur_leader = jnp.argmax(jnp.linalg.norm(goals - agent_states, axis=-1))
            graph = jit_leader_follower_assign(graph, cur_leader, goals[cur_leader])
            prompt_time_gap += 1
    else:
        if use_leader_mode: # and kk - last_leader_call > keep_mode:
            if leader_model == 'fixed':
                print('Fixed leader for step: ', kk)
                leader_id = 0
                leader_dir = goals[leader_id] - agent_states[leader_id]
                graph = jit_leader_follower_assign(graph, leader_id, leader_dir + agent_states[leader_id])
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write('Fixed leader at step: ')
                    f.write(str(kk))
                    f.write('\n')
                num_LLM_calls += 1
                prompt_time_gap = 0
            elif leader_model == 'random':
                print('Random leader for step: ', kk)
                leader_id = np.random.randint(num_agents)
                leader_dir = goals[leader_id] - agent_states[leader_id]
                graph = jit_leader_follower_assign(graph, leader_id, leader_dir + agent_states[leader_id])
                with open(f"{log_dir}/log_message.txt", "a") as f:
                    f.write('Random leader at step: ')
                    f.write(str(kk))
                    f.write('\n')
                num_LLM_calls += 1
                prompt_time_gap = 0
            else:
                if leader_assign_count < num_runtime_incontext_prompts or leader_model is None: # and not use_pre_incontext_prompts:
                    print('Hand-designed leader for step: ', kk)
                    user_prompt = create_user_prompt_fn(graph, num_agents, kk)
                    # print('search at step ', kk)
                    # use the search algorithm to select the leader
                    # graph, t_mode = jit_leader_control(graph, t_mode, keep_mode)
                    graph, t_mode = leader_control(policy, graph, t_mode, keep_mode)
                    leader_id, dir = get_leader_id_dir(graph)
                    assistant_prompt = create_assistant_prompt_fn(leader_id, dir, graph, num_agents)
                    leader_assign_count += 1
                    prompt.append(user_prompt)
                    prompt.append(assistant_prompt)
                    prompt_time_gap = 0
                    with open(f"{log_dir}/log_message.txt", "a") as f:
                        f.write('Hand-designed leader at step: ')
                        f.write(str(kk))
                        f.write('\n')
                        f.write(str(user_prompt))
                        f.write('\n')
                        f.write(str(assistant_prompt))
                        f.write('\n')
                    num_LLM_calls += 1
                else:
                    # query LLM to get the leader
                    print('LLM for step: ', kk)
                    user_prompt = create_user_prompt_fn(graph, num_agents, kk)
                    LLM_prompt = np.array(prompt).copy()
                    LLM_prompt = LLM_prompt.tolist()
                    LLM_prompt.append(user_prompt)
                    all_prompts.append(user_prompt)
                    # prompt.append(user_prompt)
                    # sleep(5)
                    with open(f"{log_dir}/log_message.txt", "a") as f:
                        f.write('LLM queried at step: ')
                        f.write(str(kk))
                        f.write('\n')
                        f.write(str(user_prompt))
                        f.write('\n')
                    if 'gpt' in leader_model:
                        mess_response=[]
                        lead_response=[]
                        dir_response=[]
                        similar_messages = 0
                        for LLM_call_iter in range(LLM_calls):
                            print('LLM call iter: ', LLM_call_iter)
                            mess_rep, response = get_response(LLM_prompt, leader_model, LLM_calls)
                            lead_i, dir_i = LLM_leader_dir_fn(mess_rep, graph, num_agents)
                            # if LLM_call_iter > 0:
                            #     # check if the lead_i and dir_i are similar to any of the previous responses
                            #     for i in range(LLM_call_iter):
                            #         if lead_i == lead_response[i] and (dir_i == dir_response[i]).all():
                            #             similar_messages += 1
                            lead_response.append(lead_i)
                            dir_response.append(dir_i)
                            mess_response.append(mess_rep)
                            sent_token_count += response.usage.prompt_tokens
                            received_token_count += response.usage.completion_tokens
                            with open(f"{log_dir}/log_message.txt", "a") as f:
                                f.write(str(response))
                                f.write('\n')
                            # if similar_messages >= LLM_calls // 2:
                            #     break
                        leader, leader_dir = find_median(mess_response, graph, num_agents)
                    else:
                        message_response, response = get_response(LLM_prompt, leader_model)
                        sent_token_count += response.usage.prompt_tokens
                        received_token_count += response.usage.completion_tokens
                        leader, leader_dir = LLM_leader_dir_fn(message_response, graph, num_agents)
                        with open(f"{log_dir}/log_message.txt", "a") as f:
                            f.write(str(response))
                            f.write('\n')
                    num_LLM_calls += 1
                    # leader = leader -1
                    if leader < 0 or leader >= num_agents:
                        print('Invalid leader id: ', leader)
                        print('STUPID LLM')
                        print('Not using the assigned leader')
                    else:
                        assistant_prompt = create_assistant_prompt_fn(leader, leader_dir, graph, num_agents)
                        # graph = jit_leader_follower_assign(graph, leader, leader_dir + agent_states[leader])
                        graph = jit_leader_follower_assign(graph, leader, leader_dir + agent_states[leader])
                        # prompt.append(user_prompt)
                        # prompt.append(assistant_prompt)
                        all_prompts.append(assistant_prompt)
                    prompt_time_gap = 0
                    
        else:
            if in_leader_mode:
                # reset the graph to the goal reaching graph
                graph = reset_graph(graph)
    return graph, num_LLM_calls, prompt, all_prompts, leader_assign_count, t_mode, prompt_time_gap, sent_token_count, received_token_count
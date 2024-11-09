import numpy as np
import alfworld.agents.environment as environment
import alfworld.agents.modules.generic as generic
import yaml
import os
import json
import re
from openai import OpenAI
from typing import Optional

port = 8003
client = OpenAI(
    base_url=f"http://localhost:{port}/v1",
    api_key="token-abc123"  # Replace with actual token if needed
)

action_templates = {
    "GotoLocation": "go to {0}",
    "OpenObject": "open {0}",
    "CloseObject": "close {0}",
    "PickupObject": "take {0} from {1}",
    "PickupObjectFromReceptacleObject": "take {0} from {1}",
    "PickupEmptyReceptacleObject": "take {0} from {1}",
    "PickupFullReceptacleObject": "take {0} from {1}",
    "PutObject": "put {0} in/on {1}",
    "PutObjectInReceptacleObject": "put {0} into {1}",
    "PutEmptyReceptacleObjectinReceptacle": "put {0} in/on {1}",
    "PutFullReceptacleObjectInReceptacle": "put {0} in {1}",
    "inventory": "inventory",
    "examineReceptacle": "examine {0}",
    "examineObject": "examine {0}",
    "ToggleObject": "use {0}",
    "HeatObject": "heat {0}",
    "CleanObject": "clean {0}",
    "CoolObject": "cool {0}",
    "SliceObject": "slice {0}",
    "look": "look"
}

def get_traj(json_file):
    traj_list = []
    # load json data
    with open(json_file) as f:
        traj_data = json.load(f)

    for ll_action in traj_data['plan']['high_pddl']:
        traj_discrete_action = ll_action['discrete_action']
        action_name, action_args = traj_discrete_action['action'], traj_discrete_action['args']
        
        # Return trajectory if action is 'NoOp'
        if action_name == 'NoOp':
            return traj_list
        
        # Retrieve template for the action_name
        action_template = action_templates.get(action_name)
        
        # Format the template if available, else create a generic action string
        if action_template:
            try:
                final_action_str = action_template.format(*action_args)
            except IndexError:
                # If we do not have enough or correct arguments to format the action template
                final_action_str = f'{action_name} ' + ' '.join(action_args)
        else:
            # Fall back to a generic representation if no template is available
            final_action_str = f'{action_name} ' + ' '.join(action_args)
        
        # Append the formatted action to the trajectory list
        traj_list.append(final_action_str)
    
    return traj_list

os.environ["ALFWORLD_DATA"] = "/home/zxia545/_Code/tony_fork_repos/alfworld/data"

base_config = '/home/zxia545/_Code/tony_fork_repos/alfworld/configs/tony_config.yaml'

save_json_dir = 'traj_result_list_3000.json'


def get_prompt(prompt):
    system_prompt = "You are a helpful assistant."
    message = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt}
    ]
    return message

def extract_action(text: str) -> Optional[str]:
    """
    Extract the real action from GPT response enclosed by 'real_action:'.
    
    Args:
        text (str): The text containing the action.
        
    Returns:
        Optional[str]: Extracted action, or None if no action is found.
    """
    pattern = r"real_action:\s*(.*)"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

def extract_action_with_retry(prompt, max_retry=3):
    completion = client.chat.completions.create(
        # model="gpt-4o-mini",
        model="gpt-4o-2024-08-06",
        messages=prompt,
        max_tokens=50,
        temperature=0.2,
        top_p=0.9
    )
    
    return_message = completion.choices[0].message.content
    action = extract_action(return_message)
    if action is None and max_retry > 0:
        return extract_action_with_retry(prompt, max_retry - 1)
    # return the action if it is not None
    return "look" if action is None else action


def check_current_task_can_by_pass(this_task_traj_item):
    
    load_json = None
    if os.path.exists(save_json_dir):
        load_json = json.load(open(save_json_dir, 'r'))
        
    this_task_path = this_task_traj_item['path_to_the_traj_folder']
    for saved_task in load_json:
        if saved_task['path_to_the_traj_folder'] == this_task_path:
            if saved_task['got_optimal_plan'] == True:
                return True, load_json
            else:
                # remove the saved task if it is not successful
                load_json.remove(saved_task)
                
                # with open(save_json_dir, 'w') as f:
                #     json.dump(load_json, f, indent=4)
    return False, load_json


# load config
with open(base_config) as reader:
    config = yaml.safe_load(reader)
env_type = config['env']['type'] # 'AlfredTWEnv' or 'AlfredThorEnv' or 'AlfredHybrid'

# setup environment
env = getattr(environment, env_type)(config, train_eval='train')
number_of_task = env.num_games

env = env.init_env(batch_size=1)

traj_result_list = []


if os.path.exists(save_json_dir):
    load_json = json.load(open(save_json_dir, 'r'))
    traj_result_list = load_json

def get_step_prompt(goal, optimal_plan, previous_action, previous_observation, current_observation, admissible_actions):
    prompt = f"Your task is to decide on the appropriate action to take next, based on the current observation, the previous action and observation, and the provided optimal plan action list.\n\n"

    prompt += f"Your goal is: {goal}\n\n"

    prompt += f"Optimal Plan Action List:\n"
    for i, plan_action in enumerate(optimal_plan, 1):
        prompt += f"{i}. {plan_action}\n"

    prompt += "\n"

    if previous_action is not None and previous_observation is not None:
        i = 1
        for pre_action, pre_observation in zip(previous_action, previous_observation):
            prompt += f"Previous step {i} Action: {pre_action}\n"
            prompt += f"Previous step {i} Observation: {pre_observation}\n"
            i += 1
    else:
        prompt += "This is the first step.\n\n"

    prompt += f"Current Observation: {current_observation}\n\n"

    prompt += f"Here is the list of admissible actions that you action must choose between one of it:\n"
    for i, admissible_action in enumerate(admissible_actions, 1):
        prompt += f"{i}. {admissible_action}\n"

    prompt += "\n"

    prompt += "Ensure your action is exactly one of the admissible actions and do not alter any characters from the provided list.\n\n"

    prompt += "Please provide the action in the following format: real_action: <your action>\n"

    return prompt

for task_time in range(number_of_task):
    obs, infos = env.reset()
    try:
        gamefile_location = infos.get('extra.gamefile')[0]
        # gamefile folder location
    except:
        continue
    
    this_task_traj_item = {}
    
    traj_json_local = os.path.join(os.path.dirname(gamefile_location), 'traj_data.json')
    # get parent parent folder and record the path
    path_to_the_traj_folder = os.path.relpath(os.path.dirname(gamefile_location), start=os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(gamefile_location))), '..'))
    
    this_task_traj_item['path_to_the_traj_folder'] = path_to_the_traj_folder
    
    can_bypass, new_json_list = check_current_task_can_by_pass(this_task_traj_item)
    
    if can_bypass:
        print(f'Skipping task {path_to_the_traj_folder}')
        continue
    else:
        if new_json_list:
            traj_result_list = new_json_list
    
    optimal_plan = get_traj(traj_json_local)
    this_task_traj_item['optimal_plan_from_traj_json'] = optimal_plan

    obs_list = []
    action_list = []

    # Initial observation
    inital_obs = obs[0]
    def get_current_goal(observation):
        task_pattern = r"Your task is to: (.*)"
        match = re.search(task_pattern, observation)
        if match:
            return match.group(1).strip()
        return None

    current_goal = get_current_goal(inital_obs)

    print(f'Initial Observation: {inital_obs}')
    
    this_task_traj_item["goal"] = current_goal
    this_task_traj_item["state_0"] = inital_obs

    previous_action = []
    previous_observation = []

    max_steps = 20

    for step in range(1, max_steps + 1):
        print(f'New Step: {step} \n\n')
        admissible_commands_list = infos['admissible_commands'][0]
        
        # Filter out 'examine' actions
        admissible_list = [cmd for cmd in admissible_commands_list if "examine" not in cmd]

        prompt_content = get_step_prompt(
            goal=current_goal,
            optimal_plan=optimal_plan,
            previous_action=previous_action,
            previous_observation=previous_observation,
            current_observation=obs[0],
            admissible_actions=admissible_list
        )
        prompt = get_prompt(prompt_content)
        
        llm_action = extract_action_with_retry(prompt)
        # Append the GPT action to the action list
        action_list.append(llm_action)
        
        # Step the environment with the current action
        obs, scores, dones, infos = env.step([llm_action])

        # Append the observation to the observation list
        obs_list.append(obs[0])
        
        this_task_traj_item[f"action_{step}"] = llm_action
        this_task_traj_item[f"state_{step}"] = obs[0]
        
        # Print the action and the resulting observation
        print("Action: {}, Obs: {}".format(llm_action, obs[0]))
        print("Scores: {}, Dones: {}".format(scores, dones))
        
        # If the task is complete, exit the loop
        if dones[0]:
            break

        # Update previous action and observation
        previous_action.append(llm_action)
        previous_observation.append(obs[0])

    # Print the optimal plan and the executed actions
    print(f'Success or Failure: {dones[0]}')
    print("Optimal Plan: ", optimal_plan)
    print("Action List: ", action_list)
    print("Observation List: ", obs_list)
    
    if dones[0]:
        this_task_traj_item['got_optimal_plan'] = True
    else:
        this_task_traj_item['got_optimal_plan'] = False
    
    traj_result_list.append(this_task_traj_item)
    
    # Save the result when task is done or max steps reached
    with open(save_json_dir, 'w') as f:
        json.dump(traj_result_list, f, indent=4)

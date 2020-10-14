import os
import sys
sys.path.insert(0, os.environ['ALFRED_ROOT'])

import json
import argparse

from env.thor_env import ThorEnv
from agents.detector.mrcnn import load_pretrained_model
from agents.controller import OracleAgent, OracleAStarAgent, MaskRCNNAgent, MaskRCNNAStarAgent


def setup_scene(env, traj_data, r_idx, args, reward_type='dense'):
    '''
    intialize the scene and agent from the task info
    '''
    # scene setup
    scene_num = traj_data['scene']['scene_num']
    object_poses = traj_data['scene']['object_poses']
    dirty_and_empty = traj_data['scene']['dirty_and_empty']
    object_toggles = traj_data['scene']['object_toggles']

    scene_name = 'FloorPlan%d' % scene_num
    env.reset(scene_name)
    env.restore_scene(object_poses, object_toggles, dirty_and_empty)

    # initialize to start position
    env.step(dict(traj_data['scene']['init_action']))

    # print goal instr
    print("Task: %s" % (traj_data['turk_annotations']['anns'][r_idx]['task_desc']))

    # setup task for reward
    env.set_task(traj_data, args, reward_type=reward_type)


def main(args):

    env = ThorEnv()

    root = args.problem
    json_file = os.path.join(root, 'traj_data.json')
    with open(json_file, 'r') as f:
        traj_data = json.load(f)

    setup_scene(env, traj_data, 0, args)

    if args.controller == "oracle":
        AgentModule = OracleAgent
        agent = AgentModule(env, traj_data, traj_root=root, load_receps=args.load_receps, debug=args.debug)
    elif args.controller == "oracle_astar":
        AgentModule = OracleAStarAgent
        agent = AgentModule(env, traj_data, traj_root=root, load_receps=args.load_receps, debug=args.debug)
    elif args.controller == "mrcnn":
        AgentModule = MaskRCNNAgent
        mask_rcnn = load_pretrained_model('./detector/data/400_scenes/mcrnn_alfred_004.pth')
        agent = AgentModule(env, traj_data, traj_root=root,
                            pretrained_model=mask_rcnn,
                            load_receps=args.load_receps, debug=args.debug)
    elif args.controller == "mrcnn_astar":
        AgentModule = MaskRCNNAStarAgent
        mask_rcnn = load_pretrained_model('./detector/data/400_scenes/mcrnn_alfred_004.pth')
        agent = AgentModule(env, traj_data, traj_root=root,
                            pretrained_model=mask_rcnn,
                            load_receps=args.load_receps, debug=args.debug)
    else:
        raise NotImplementedError()

    print(agent.feedback)
    while True:
        cmd = input()
        agent.step(cmd)
        print(agent.feedback)
        # print (agent.get_admissible_commands())

        done = env.get_goal_satisfied()
        if done:
            print("You won!")
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("problem", help="Path to folder containing pddl and traj_data files")
    parser.add_argument("--controller", default="oracle", choices=["oracle", "oracle_astar", "mrcnn", "mrcnn_astar"])
    parser.add_argument("--debug", action="store_true")
    parser.add_argument('--load_receps', action="store_true")
    parser.add_argument('--reward_config', type=str, default="agents/config/rewards.json")
    args = parser.parse_args()

    main(args)




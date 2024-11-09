import copy
import json
import os
from abc import ABC

import alfworld.gen.constants as constants
from alfworld.gen.game_states.game_state_base import GameStateBase
from alfworld.gen.planner import ff_planner_handler
from alfworld.gen.utils import game_util
from alfworld.gen.utils import py_util


class PlannedGameState(GameStateBase, ABC):
    @staticmethod
    def fix_pddl_str_chars(input_str):
        return py_util.multireplace(input_str,
                                    {'-': '_minus_',
                                     '#': '-',
                                     '|': '_bar_',
                                     '+': '_plus_',
                                     '.': '_dot_',
                                     ',': '_comma_'})

    action_space = [
        {'action': 'Explore'},
        {'action': 'Scan'},
        {'action': 'Plan'},
        {'action': 'End'},
    ]

    action_to_ind = {action['action']: ii for ii, action in enumerate(action_space)}

    def __init__(self, env, seed=None, action_space=None, domain=None, domain_path=None):
        super(PlannedGameState, self).__init__(env, seed, action_space)
        self.planner = ff_planner_handler.PlanParser(domain_path)
        self.domain = domain
        self.terminal = False
        self.problem_id = -1
        self.in_receptacle_ids = {}
        self.was_in_receptacle_ids = {}
        self.need_plan_update = True
        self.pddl_start = None
        self.pddl_init = None
        self.pddl_goal = None
        self.scene_seed = None
        self.object_target = None
        self.parent_target = None
        self.receptacle_to_point = {}
        self.point_to_receptacle = {}
        self.object_to_point = {}
        self.point_to_object = {}
        self.plan = None
        self.next_action = None
        self.failed_plan_action = False
        self.placed_items = set()
        self.openable_object_to_point = None

    def get_goal_pddl(self):
        raise NotImplementedError

    def state_to_pddl(self):
        object_dict = game_util.get_object_dict(self.env.last_event.metadata)
        if self.pddl_start is None:
            # Not previously written to file
            self.planner.problem_id = self.problem_id
            receptacle_types = copy.deepcopy(constants.RECEPTACLES) - set(constants.MOVABLE_RECEPTACLES)
            objects = copy.deepcopy(constants.OBJECTS_SET) - receptacle_types
            object_str = '\n        '.join([obj + ' # object' for obj in objects])

            self.knife_obj = {'ButterKnife', 'Knife'} if constants.data_dict['pddl_params']['object_sliced'] else {}

            otype_str = '\n        '.join([obj + 'Type # otype' for obj in objects])
            rtype_str = '\n        '.join([obj + 'Type # rtype' for obj in receptacle_types])

            self.pddl_goal = self.get_goal_pddl()

            self.pddl_start = '''
(define (problem plan_%s)
    (:domain %s)
    (:metric minimize (totalCost))
    (:objects
        agent1 # agent
        %s
        %s
        %s
''' % (
                self.problem_id,
                self.domain,

                object_str,
                otype_str,
                rtype_str,
                )

            self.pddl_init = '''
    (:init
        (= (totalCost) 0)
'''

            self.pddl_start = PlannedGameState.fix_pddl_str_chars(self.pddl_start)
            self.pddl_init = PlannedGameState.fix_pddl_str_chars(self.pddl_init)
            self.pddl_goal = PlannedGameState.fix_pddl_str_chars(self.pddl_goal)
            
        # pddl_mid section
        agent_location = 'loc|%d|%d|%d|%d' % (self.pose[0], self.pose[1], self.pose[2], self.pose[3])

        agent_location_str = '\n        (atLocation agent1 %s)' % agent_location
        opened_receptacle_str = '\n        '.join(['(opened %s)' % obj
                                                   for obj in self.currently_opened_object_ids])

        movable_recep_cls_with_knife = []
        in_receptacle_strs = []
        was_in_receptacle_strs = []
        for key, val in self.in_receptacle_ids.items():
            if len(val) == 0:
                continue
            key_cls = object_dict[key]['objectType']
            if key_cls in constants.MOVABLE_RECEPTACLES_SET:
                recep_str = 'inReceptacleObject'
            else:
                recep_str = 'inReceptacle'
            for vv in val:
                vv_cls = object_dict[vv]['objectType']
                if (vv_cls == constants.OBJECTS[self.object_target] or
                        (self.mrecep_target is not None and vv_cls == constants.OBJECTS[self.mrecep_target]) or
                        (len(self.knife_obj) > 0 and vv_cls in self.knife_obj)):

                    # if knife is inside a movable receptacle, make sure to add it to the object list
                    if recep_str == 'inReceptacleObject':
                        movable_recep_cls_with_knife.append(key_cls)

                    in_receptacle_strs.append('(%s %s %s)' % (
                        recep_str,
                        vv,
                        key)
                    )
                if key_cls not in constants.MOVABLE_RECEPTACLES_SET and vv_cls == constants.OBJECTS[self.object_target]:
                    was_in_receptacle_strs.append('(wasInReceptacle  %s %s)' % (vv, key))

        in_receptacle_str = '\n        '.join(in_receptacle_strs)
        was_in_receptacle_str = '\n        '.join(was_in_receptacle_strs)

        # Note which openable receptacles we can safely open (precomputed).
        openable_objects = self.openable_object_to_point.keys()

        metadata_objects = self.env.last_event.metadata['objects']
        receptacles = set({obj['objectId'] for obj in metadata_objects
                           if obj['objectType'] in constants.RECEPTACLES and obj['objectType'] not in constants.MOVABLE_RECEPTACLES_SET})

        objects = set({obj['objectId'] for obj in metadata_objects if
                       (obj['objectType'] == constants.OBJECTS[self.object_target]
                        or obj['objectType'] in constants.MOVABLE_RECEPTACLES_SET
                        or (self.mrecep_target is not None and obj['objectType'] == constants.OBJECTS[self.mrecep_target])
                        or ((self.toggle_target is not None and obj['objectType'] == constants.OBJECTS[self.toggle_target])
                            or ((len(self.knife_obj) > 0 and
                                 (obj['objectType'] in self.knife_obj or
                                  obj['objectType'] in movable_recep_cls_with_knife)))))})

        if len(self.inventory_ids) > 0:
            objects = objects | self.inventory_ids
        if len(self.placed_items) > 0:
            objects = objects | self.placed_items

        receptacle_str = '\n        '.join(sorted([receptacle + ' # receptacle'
                                            for receptacle in receptacles]))

        object_str = '\n        '.join(sorted([obj + ' # object' for obj in objects]))

        locations = set()
        for key, val in self.receptacle_to_point.items():
            key_cls = object_dict[key]['objectType']
            if key_cls not in constants.MOVABLE_RECEPTACLES_SET:
                locations.add(tuple(val.tolist()))
        for obj, loc in self.object_to_point.items():
            obj_cls = object_dict[obj]['objectType']
            if (obj_cls == constants.OBJECTS[self.object_target] or
                    (self.toggle_target is not None and obj_cls == constants.OBJECTS[self.toggle_target]) or
                    (len(self.knife_obj) > 0 and obj_cls in self.knife_obj) or
                    (obj_cls in constants.MOVABLE_RECEPTACLES_SET)):
                locations.add(tuple(loc))

        location_str = ('\n        '.join(['loc|%d|%d|%d|%d # location' % (*loc,)
                                          for loc in locations]) +
                        '\n        %s # location' % agent_location)

        if constants.PRUNE_UNREACHABLE_POINTS:
            # don't flag problematic receptacleTypes for the planner.
            receptacle_type_str = '\n        '.join(['(receptacleType %s %sType)' % (
                receptacle, object_dict[receptacle]['objectType']) for receptacle in receptacles
                                                     if object_dict[receptacle]['objectType'] not in constants.OPENABLE_CLASS_SET or
                                                        receptacle in openable_objects])
        else:
            receptacle_type_str = '\n        '.join(['(receptacleType %s %sType)' % (
                receptacle, object_dict[receptacle]['objectType']) for receptacle in receptacles])

        object_type_str = '\n        '.join(['(objectType %s %sType)' % (
            obj, object_dict[obj]['objectType']) for obj in objects])

        receptacle_objects_str = '\n        '.join(['(isReceptacleObject %s)' % (
            obj) for obj in objects if object_dict[obj]['objectType'] in constants.MOVABLE_RECEPTACLES_SET])

        if constants.PRUNE_UNREACHABLE_POINTS:
            openable_str = '\n        '.join(['(openable %s)' % receptacle for receptacle in receptacles
                                              if object_dict[receptacle]['objectType'] in constants.OPENABLE_CLASS_SET])
        else:
            # don't flag problematic open objects as openable for the planner.
            openable_str = '\n        '.join(['(openable %s)' % receptacle for receptacle in receptacles
                                              if object_dict[receptacle]['objectType'] in constants.OPENABLE_CLASS_SET and
                                              receptacle in openable_objects])

        dists = []
        dist_points = list(locations | {(self.pose[0], self.pose[1], self.pose[2], self.pose[3])})
        for dd, l_start in enumerate(dist_points[:-1]):
            for l_end in dist_points[dd + 1:]:
                actions, path = self.gt_graph.get_shortest_path_unweighted(l_start, l_end)
                # Should cost one more for the trouble of going there at all. Discourages waypoints.
                dist = len(actions) + 1
                dists.append('(= (distance loc|%d|%d|%d|%d loc|%d|%d|%d|%d) %d)' % (
                    l_start[0], l_start[1], l_start[2], l_start[3],
                    l_end[0], l_end[1], l_end[2], l_end[3], dist))
                dists.append('(= (distance loc|%d|%d|%d|%d loc|%d|%d|%d|%d) %d)' % (
                    l_end[0], l_end[1], l_end[2], l_end[3],
                    l_start[0], l_start[1], l_start[2], l_start[3], dist))
        location_distance_str = '\n        '.join(dists)


        # clean objects
        cleanable_str = '\n        '.join(['(cleanable %s)' % obj for obj in objects
                                          if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Cleanable']])

        is_clean_str = '\n        '.join((['(isClean %s)' % obj
                                           for obj in self.cleaned_object_ids if object_dict[obj]['objectType'] == constants.OBJECTS[self.object_target]]))

        # heat objects
        heatable_str = '\n        '.join(['(heatable %s)' % obj for obj in objects
                                          if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Heatable']])

        is_hot_str = '\n        '.join((['(isHot %s)' % obj
                                         for obj in self.hot_object_ids if object_dict[obj]['objectType'] == constants.OBJECTS[self.object_target]]))

        # cool objects
        coolable_str = '\n        '.join(['(coolable %s)' % obj for obj in objects
                                          if object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Coolable']])

        # toggleable objects
        toggleable_str = '\n        '.join(['(toggleable %s)' % obj
                                            for obj in self.toggleable_object_ids
                                            if (self.toggle_target is not None
                                                and object_dict[obj]['objectType'] == constants.OBJECTS[self.toggle_target])])

        is_on_str = '\n        '.join(['(isOn %s)' % obj
                                       for obj in self.on_object_ids
                                       if (self.toggle_target is not None
                                           and object_dict[obj]['objectType'] == constants.OBJECTS[self.toggle_target])])

        # sliceable objects
        sliceable_str = '\n        '.join(['(sliceable %s)' % obj for obj in objects
                                          if (object_dict[obj]['objectType'] in constants.VAL_ACTION_OBJECTS['Sliceable'])])

        # sliced objects
        # TODO cleanup: sliced_object_ids is never added to. Does that matter?
        is_sliced_str = '\n        '.join((['(isSliced %s)' % obj
                                            for obj in self.sliced_object_ids
                                            if object_dict[obj]['objectType'] == constants.OBJECTS[self.object_target]]))

        # look for objects that are already cool
        for (key, val) in self.was_in_receptacle_ids.items():
            if 'Fridge' in key:
                for vv in val:
                    self.cool_object_ids.add(vv)

        is_cool_str = '\n        '.join((['(isCool %s)' % obj
                                          for obj in self.cool_object_ids if object_dict[obj]['objectType'] == constants.OBJECTS[self.object_target]]))

        # Receptacle Objects
        recep_obj_str = '\n        '.join(['(isReceptacleObject %s)' % obj for obj in receptacles
                                          if (object_dict[obj]['objectType'] in constants.MOVABLE_RECEPTACLES_SET and
                                              (self.mrecep_target is not None and object_dict[obj]['objectType'] == constants.OBJECTS[self.mrecep_target]))])

        receptacle_nearest_point_strs = sorted(
            ['(receptacleAtLocation %s loc|%d|%d|%d|%d)' % (obj_id, *point)
             for obj_id, point in self.receptacle_to_point.items()
             if (object_dict[obj_id]['objectType'] in constants.RECEPTACLES and
                 object_dict[obj_id]['objectType'] not in constants.MOVABLE_RECEPTACLES_SET)
             ])
        receptacle_at_location_str = '\n        '.join(receptacle_nearest_point_strs)
        extra_facts = self.get_extra_facts()

        pddl_mid_start = '''
        %s
        %s
        %s
        )
''' % (
            object_str,
            receptacle_str,
            location_str,
            )
        pddl_mid_init = '''
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        %s
        )
''' % (
            receptacle_type_str,
            object_type_str,
            receptacle_objects_str,
            openable_str,
            agent_location_str,
            opened_receptacle_str,
            cleanable_str,
            is_clean_str,
            heatable_str,
            coolable_str,
            is_hot_str,
            is_cool_str,
            toggleable_str,
            is_on_str,
            recep_obj_str,
            sliceable_str,
            is_sliced_str,
            in_receptacle_str,
            was_in_receptacle_str,
            location_distance_str,
            receptacle_at_location_str,
            extra_facts,
            )

        receptacle_type_str = PlannedGameState.fix_pddl_str_chars(receptacle_type_str)
        object_type_str = PlannedGameState.fix_pddl_str_chars(object_type_str)
        receptacle_objects_str = PlannedGameState.fix_pddl_str_chars(receptacle_objects_str)
        openable_str = PlannedGameState.fix_pddl_str_chars(openable_str)
        agent_location_str = PlannedGameState.fix_pddl_str_chars(agent_location_str)
        opened_receptacle_str = PlannedGameState.fix_pddl_str_chars(opened_receptacle_str)
        cleanable_str = PlannedGameState.fix_pddl_str_chars(cleanable_str)
        is_clean_str = PlannedGameState.fix_pddl_str_chars(is_clean_str)
        heatable_str = PlannedGameState.fix_pddl_str_chars(heatable_str)
        coolable_str = PlannedGameState.fix_pddl_str_chars(coolable_str)
        is_hot_str = PlannedGameState.fix_pddl_str_chars(is_hot_str)
        is_cool_str = PlannedGameState.fix_pddl_str_chars(is_cool_str)
        toggleable_str = PlannedGameState.fix_pddl_str_chars(toggleable_str)
        is_on_str = PlannedGameState.fix_pddl_str_chars(is_on_str)
        recep_obj_str = PlannedGameState.fix_pddl_str_chars(recep_obj_str)
        sliceable_str = PlannedGameState.fix_pddl_str_chars(sliceable_str)
        is_sliced_str = PlannedGameState.fix_pddl_str_chars(is_sliced_str)
        in_receptacle_str = PlannedGameState.fix_pddl_str_chars(in_receptacle_str)
        was_in_receptacle_str = PlannedGameState.fix_pddl_str_chars(was_in_receptacle_str)
        location_distance_str = PlannedGameState.fix_pddl_str_chars(location_distance_str)
        receptacle_at_location_str = PlannedGameState.fix_pddl_str_chars(receptacle_at_location_str)
        extra_facts = PlannedGameState.fix_pddl_str_chars(extra_facts)
        
        
        
        
        split_txt = '\n        '
        mid_receptacle_type_objects = receptacle_type_str.split(split_txt)
        mid_object_type_objects = object_type_str.split(split_txt)
        mid_receptacle_objects = receptacle_objects_str.split(split_txt)
        mid_openable_objects = openable_str.split(split_txt)
        mid_agent_location = agent_location_str.split(split_txt)
        mid_opened_receptacle = opened_receptacle_str.split(split_txt)
        mid_cleanable_objects = cleanable_str.split(split_txt)
        mid_is_clean_objects = is_clean_str.split(split_txt)
        mid_heatable_objects = heatable_str.split(split_txt)
        mid_coolable_objects = coolable_str.split(split_txt)
        mid_is_hot_objects = is_hot_str.split(split_txt)
        mid_is_cool_objects = is_cool_str.split(split_txt)
        mid_toggleable_objects = toggleable_str.split(split_txt)
        mid_is_on_objects = is_on_str.split(split_txt)
        mid_recep_obj_objects = recep_obj_str.split(split_txt)
        mid_sliceable_objects = sliceable_str.split(split_txt)
        mid_is_sliced_objects = is_sliced_str.split(split_txt)
        mid_in_receptacle_objects = in_receptacle_str.split(split_txt)
        mid_was_in_receptacle_objects = was_in_receptacle_str.split(split_txt)
        mid_location_distance_objects = location_distance_str.split(split_txt)
        mid_receptacle_at_location_objects = receptacle_at_location_str.split(split_txt)
        mid_extra_facts = extra_facts.split(split_txt)
        
        
        mid_location_list = []
        mid_object_list = []
        mid_rtype_list = []
        mid_receptacle_list = []
        mid_otype_list = []
        
        # get mid_receptacle_type_objects
        if len(mid_receptacle_type_objects) > 0:
            for this_receptacle_type in mid_receptacle_type_objects:
                # example is like this :(receptacleType DiningTable_bar__minus_03_dot_03_bar__plus_00_dot_00_bar__minus_00_dot_44 DiningTableType)
                this_receptacle_type = this_receptacle_type.replace('(', '').replace(')', '')
                
                parts = this_receptacle_type.split(' ')
                if len(parts) >= 3:
                    this_receptacle = parts[1]
                    this_rtype = parts[2]

                    mid_receptacle_list.append(this_receptacle)
                    mid_rtype_list.append(this_rtype)
                    
        # get mid_object_type_objects
        if len(mid_object_type_objects) > 0:
            for this_object_type in mid_object_type_objects:
                # example is like this :(objectType Apple_bar__plus_00_dot_00_bar__minus_00_dot_44 AppleType)
                this_object_type = this_object_type.replace('(', '').replace(')', '')
                
                parts = this_object_type.split(' ')
                if len(parts) >= 3:
                    this_object = parts[1]
                    this_otype = parts[2]

                    mid_object_list.append(this_object)
                    mid_otype_list.append(this_otype)
        
        if len(mid_receptacle_objects) > 0:
            for this_receptacle in mid_receptacle_objects:
                this_receptacle = this_receptacle.replace('(', '').replace(')', '')
                parts = this_receptacle.split(' ')
                if len(parts) >= 2:
                    this_object =parts[1]
                    mid_object_list.append(this_object)

                
        
        # Example parsing mid_openable_objects
        if len(mid_openable_objects) > 0:
            for openable in mid_openable_objects:
                openable = openable.replace('(', '').replace(')', '')
                parts = openable.split(' ')
                if len(parts) >= 2:
                    openable_obj = parts[1]
                    mid_receptacle_list.append(openable_obj)
                    # print(f"Openable Object: {openable_obj}")

        # Example parsing mid_agent_location
        if len(mid_agent_location) > 0:
            for location in mid_agent_location:
                location = location.replace('(', '').replace(')', '')
                parts = location.split(' ')
                if len(parts) >= 3:
                    agent = parts[1]
                    loc = parts[2]
                    mid_location_list.append(loc)
                    # print(f"Agent: {agent}, Location: {loc}")

        # Example parsing mid_opened_receptacle
        if len(mid_opened_receptacle) > 0:
            for opened in mid_opened_receptacle:
                opened = opened.replace('(', '').replace(')', '')
                parts = opened.split(' ')
                if len(parts) >= 2:
                    opened_recep = parts[1]
                    mid_receptacle_list.append(opened_recep)
                    # print(f"Opened Receptacle: {opened_recep}")

        # Example parsing mid_cleanable_objects
        if len(mid_cleanable_objects) > 0:
            for cleanable in mid_cleanable_objects:
                cleanable = cleanable.replace('(', '').replace(')', '')
                parts = cleanable.split(' ')
                if len(parts) >= 2:
                    cleanable_obj = parts[1]
                    mid_object_list.append(cleanable_obj)
                    # print(f"Cleanable Object: {cleanable_obj}")

        # Example parsing mid_is_clean_objects
        if len(mid_is_clean_objects) > 0:
            for clean in mid_is_clean_objects:
                clean = clean.replace('(', '').replace(')', '')
                parts = clean.split(' ')
                if len(parts) >= 2:
                    clean_obj = parts[1]
                    mid_object_list.append(clean_obj)
                    # print(f"Is Clean Object: {clean_obj}")

        # Example parsing mid_heatable_objects
        if len(mid_heatable_objects) > 0:
            for heatable in mid_heatable_objects:
                heatable = heatable.replace('(', '').replace(')', '')
                parts = heatable.split(' ')
                if len(parts) >= 2:
                    heatable_obj = parts[1]
                    mid_object_list.append(heatable_obj)
                    # print(f"Heatable Object: {heatable_obj}")

        # Example parsing mid_coolable_objects
        if len(mid_coolable_objects) > 0:
            for coolable in mid_coolable_objects:
                coolable = coolable.replace('(', '').replace(')', '')
                parts = coolable.split(' ')
                if len(parts) >= 2:
                    coolable_obj = parts[1]
                    mid_object_list.append(coolable_obj)
                    # print(f"Coolable Object: {coolable_obj}")

        # Example parsing mid_is_hot_objects
        if len(mid_is_hot_objects) > 0:
            for hot in mid_is_hot_objects:
                hot = hot.replace('(', '').replace(')', '')
                parts = hot.split(' ')
                if len(parts) >= 2:
                    hot_obj = parts[1]
                    mid_object_list.append(hot_obj)
                    # print(f"Is Hot Object: {hot_obj}")

        # Example parsing mid_is_cool_objects
        if len(mid_is_cool_objects) > 0:
            for cool in mid_is_cool_objects:
                cool = cool.replace('(', '').replace(')', '')
                parts = cool.split(' ')
                if len(parts) >= 2:
                    cool_obj = parts[1]
                    mid_object_list.append(cool_obj)
                    # print(f"Is Cool Object: {cool_obj}")

        # Example parsing mid_toggleable_objects
        if len(mid_toggleable_objects) > 0:
            for toggleable in mid_toggleable_objects:
                toggleable = toggleable.replace('(', '').replace(')', '')
                parts = toggleable.split(' ')
                if len(parts) >= 2:
                    toggleable_obj = parts[1]
                    mid_object_list.append(toggleable_obj)
                    # print(f"Toggleable Object: {toggleable_obj}")

        # Example parsing mid_is_on_objects
        if len(mid_is_on_objects) > 0:
            for is_on in mid_is_on_objects:
                is_on = is_on.replace('(', '').replace(')', '')
                parts = is_on.split(' ')
                if len(parts) >= 2:
                    is_on_obj = parts[1]
                    mid_object_list.append(is_on_obj)
                    # print(f"Is On Object: {is_on_obj}")

        # Example parsing mid_recep_obj_objects
        if len(mid_recep_obj_objects) > 0:
            for recep_obj in mid_recep_obj_objects:
                recep_obj = recep_obj.replace('(', '').replace(')', '')
                parts = recep_obj.split(' ')
                if len(parts) >= 2:
                    recep_obj_name = parts[1]
                    mid_object_list.append(recep_obj_name)
                    # print(f"Receptacle Object: {recep_obj_name}")

        # Example parsing mid_sliceable_objects
        if len(mid_sliceable_objects) > 0:
            for sliceable in mid_sliceable_objects:
                sliceable = sliceable.replace('(', '').replace(')', '')
                parts = sliceable.split(' ')
                if len(parts) >= 2:
                    sliceable_obj = parts[1]
                    mid_object_list.append(sliceable_obj)
                    # print(f"Sliceable Object: {sliceable_obj}")

        # Example parsing mid_is_sliced_objects
        if len(mid_is_sliced_objects) > 0:
            for is_sliced in mid_is_sliced_objects:
                is_sliced = is_sliced.replace('(', '').replace(')', '')
                parts = is_sliced.split(' ')
                if len(parts) >= 2:
                    is_sliced_obj = parts[1]
                    mid_object_list.append(is_sliced_obj)
                    # print(f"Is Sliced Object: {is_sliced_obj}")

        # Example parsing mid_in_receptacle_objects
        if len(mid_in_receptacle_objects) > 0:
            for in_receptacle in mid_in_receptacle_objects:
                in_receptacle = in_receptacle.replace('(', '').replace(')', '')
                parts = in_receptacle.split(' ')
                if len(parts) >= 3:
                    obj = parts[1]
                    recep = parts[2]
                    mid_object_list.append(obj)
                    mid_object_list.append(recep)
                    # print(f"Object: {obj}, In Receptacle: {recep}")

        # Example parsing mid_was_in_receptacle_objects
        if len(mid_was_in_receptacle_objects) > 0:
            for was_in_receptacle in mid_was_in_receptacle_objects:
                was_in_receptacle = was_in_receptacle.replace('(', '').replace(')', '')
                parts = was_in_receptacle.split(' ')
                if len(parts) >= 3:
                    obj = parts[2]
                    recep = parts[3]
                    mid_object_list.append(obj)
                    mid_receptacle_list.append(recep)
                    # print(f"Object: {obj}, Was In Receptacle: {recep}")

        # Example parsing mid_location_distance_objects
        if len(mid_location_distance_objects) > 0:
            for location_distance in mid_location_distance_objects:
                location_distance = location_distance.split('(')[2]
                location_distance = location_distance.split(')')[0]
                parts = location_distance.split(' ')
                if len(parts) >= 3:
                    loc1 = parts[1]
                    loc2 = parts[2]
                    mid_location_list.append(loc1)
                    mid_location_list.append(loc2)
                    # print(f"Location Distance: {loc1} to {loc2}")

        # Example parsing mid_receptacle_at_location_objects
        if len(mid_receptacle_at_location_objects) > 0:
            for recep_at_location in mid_receptacle_at_location_objects:
                recep_at_location = recep_at_location.replace('(', '').replace(')', '')
                parts = recep_at_location.split(' ')
                if len(parts) >= 3:
                    recep = parts[1]
                    loc = parts[2]
                    mid_receptacle_list.append(recep)
                    mid_location_list.append(loc)
                    # print(f"Receptacle: {recep}, At Location: {loc}")

        # Example parsing mid_extra_facts
        if len(mid_extra_facts) > 0:
            for fact in mid_extra_facts:
                fact = fact.replace('(', '').replace(')', '')
                parts = fact.split(' ')
                if len(parts) >= 3:
                    obj = parts[1]
                    loc = parts[2]
                    mid_object_list.append(obj)
                    mid_location_list.append(loc)
                    # print(f"Object: {obj}, Location: {loc}")
                    
        # remove duplicates
        mid_location_list = list(set(mid_location_list))
        mid_object_list = list(set(mid_object_list))
        mid_rtype_list = list(set(mid_rtype_list))
        mid_receptacle_list = list(set(mid_receptacle_list))
        
                    
        this_mid_receptacle_str = '\n        '.join(sorted([receptacle + ' - receptacle'
                                            for receptacle in mid_receptacle_list]))

        this_mid_object_str = '\n        '.join(sorted([obj + ' - object' for obj in mid_object_list]))
        
        this_mid_location_str = '\n        '.join(sorted([obj + ' - location' for obj in mid_location_list]))
        
        this_mid_otype_str = '\n        '.join(sorted([obj + ' - otype' for obj in mid_otype_list]))
        this_mid_rtype_str = '\n        '.join(sorted([obj + ' - rtype' for obj in mid_rtype_list]))
                
        own_pddl_mid_start = '''
        %s
        %s
        %s
        %s
        %s
        )
''' % (
            this_mid_object_str,
            this_mid_receptacle_str,
            this_mid_location_str,
            this_mid_otype_str,
            this_mid_rtype_str
            )
        """
            agent
            location
            receptacle
            object
            rtype
            otype
            
            
            
            
            (atLocation ?a - agent ?l - location)                     ; true if the agent is at the location
            (receptacleAtLocation ?r - receptacle ?l - location)      ; true if the receptacle is at the location (constant)
            (objectAtLocation ?o - object ?l - location)              ; true if the object is at the location
            (openable ?r - receptacle)                                ; true if a receptacle is openable
            (opened ?r - receptacle)                                  ; true if a receptacle is opened
            (inReceptacle ?o - object ?r - receptacle)                ; object ?o is in receptacle ?r
            (isReceptacleObject ?o - object)                          ; true if the object can have things put inside it
            (inReceptacleObject ?innerObject - object ?outerObject - object)                ; object ?innerObject is inside object ?outerObject
            (isReceptacleObjectFull ?o - object)                      ; true if the receptacle object contains something
            (wasInReceptacle ?o - object ?r - receptacle)             ; object ?o was or is in receptacle ?r now or some time in the past
            (checked ?r - receptacle)                                 ; whether the receptacle has been looked inside/visited
            (examined ?l - location)                                  ; TODO
            (receptacleType ?r - receptacle ?t - rtype)               ; the type of receptacle (Cabinet vs Cabinet|01|2...)
            (canContain ?rt - rtype ?ot - otype)                      ; true if receptacle can hold object
            (objectType ?o - object ?t - otype)                       ; the type of object (Apple vs Apple|01|2...)
            (holds ?a - agent ?o - object)                            ; object ?o is held by agent ?a
            (holdsAny ?a - agent)                                     ; agent ?a holds an object
            (holdsAnyReceptacleObject ?a - agent)                        ; agent ?a holds a receptacle object
            (full ?r - receptacle)                                    ; true if the receptacle has no remaining space
            (isClean ?o - object)                                     ; true if the object has been clean in sink
            (cleanable ?o - object)                                   ; true if the object can be placed in a sink
            (isHot ?o - object)                                       ; true if the object has been heated up
            (heatable ?o - object)                                    ; true if the object can be heated up in a microwave
            (isCool ?o - object)                                      ; true if the object has been cooled
            (coolable ?o - object)                                    ; true if the object can be cooled in the fridge
            (pickupable ?o - object)                                   ; true if the object can be picked up
            (moveable ?o - object)                                      ; true if the object can be moved
            (toggleable ?o - object)                                  ; true if the object can be turned on/off
            (isOn ?o - object)                                        ; true if the object is on
            (isToggled ?o - object)                                   ; true if the object has been toggled
            (sliceable ?o - object)                                   ; true if the object can be sliced
            (isSliced ?o - object) 
        """
        
        
        

        pddl_mid_start = PlannedGameState.fix_pddl_str_chars(pddl_mid_start)
        pddl_mid_init = PlannedGameState.fix_pddl_str_chars(pddl_mid_init)

        pddl_str = (self.pddl_start + '\n' +
                    own_pddl_mid_start + '\n' +
                    self.pddl_init + '\n' +
                    pddl_mid_init + '\n' +
                    self.pddl_goal)


        state_save_path = constants.save_path.replace('/raw_images', '/pddl_states')
        if not os.path.exists(state_save_path):
            os.makedirs(state_save_path)
        pddl_state_next_idx = len(constants.data_dict['pddl_state'])
        state_save_file = state_save_path + ('/problem_%s.pddl' % pddl_state_next_idx)

        with open(state_save_file, 'w') as fid:
            fid.write(pddl_str)
            fid.flush()
        constants.data_dict['pddl_state'].append('problem_%s.pddl' % pddl_state_next_idx)

        with open('%s/planner/generated_problems/problem_%s.pddl' % (self.dname, self.problem_id), 'w') as fid:
            fid.write(pddl_str)
            fid.flush()

        return pddl_str

    def get_extra_facts(self):
        raise NotImplementedError

    def get_teleport_action(self, action):
        nearest_point = tuple(map(int, action['location'].split('|')[1:]))
        next_action = {'action': 'TeleportFull',
                       'x': nearest_point[0] * constants.AGENT_STEP_SIZE,
                       'y': self.agent_height,
                       'z': nearest_point[1] * constants.AGENT_STEP_SIZE,
                       'rotateOnTeleport': True,
                       'rotation': nearest_point[2] * 90,
                       'horizon': nearest_point[3]
                       }
        return next_action

    def get_plan_action(self, action):
        if action['action'] == 'GotoLocation':
            action = self.get_teleport_action(action)
        return action

    def get_next_plan_action(self, force_update=False):
        if force_update:
            self.need_plan_update = True
        if self.need_plan_update:
            self.plan = self.get_current_plan()
            self.next_action = self.plan[0]
            if self.next_action['action'] == 'GotoLocation':
                self.next_action = self.get_teleport_action(self.next_action)
        if constants.DEBUG:
            print('\nnew plan\n' + '\n'.join(['%03d %s' % (ii, game_util.get_action_str(pl))
                                              for ii, pl in enumerate(self.plan)]), '\n')
        return self.next_action

    def get_current_plan(self, force_update=False):
        if self.failed_plan_action:
            self.plan = [{'action': 'End', 'value': 0}]
            self.failed_plan_action = False
            return self.plan
        if force_update:
            self.need_plan_update = True
        if self.need_plan_update:
            self.update_receptacle_nearest_points()
            self.plan = []
            # When there are no receptacles, there's nothing to plan.
            # Only happens if called too early (before room exploration).
            if len(self.receptacle_to_point) > 0:
                self.state_to_pddl()
                self.plan = self.planner.get_plan()
            self.need_plan_update = False
            if len(self.plan) == 0:
                # Problem is solved, plan is empty
                self.plan = [{'action': 'End', 'value': 1}]
        return self.plan

    def get_setup_info(self, info=None):
        raise NotImplementedError

    def reset(self, seed=None, info=None, scene=None, objs=None):
        if self.problem_id is not None:
            # clean up old problem
            if (not constants.EVAL and not constants.DEBUG and
                    os.path.exists('%s/planner/generated_problems/problem_%s.pddl' % (self.dname, self.problem_id))):
                os.remove('%s/planner/generated_problems/problem_%s.pddl' % (self.dname, self.problem_id))

        self.terminal = False
        self.problem_id = -1
        self.in_receptacle_ids = {}
        self.was_in_receptacle_ids = {}
        self.need_plan_update = True
        self.pddl_start = None
        self.pddl_init = None
        self.pddl_goal = None
        self.scene_seed = seed
        self.scene_num = None
        self.object_target = None
        self.parent_target = None
        self.receptacle_to_point = {}
        self.point_to_receptacle = {}
        self.object_to_point = {}
        self.point_to_object = {}
        self.plan = None
        self.failed_plan_action = False
        self.placed_items = set()

        if seed is not None:
            print('set seed in planned_game_state', seed)
            self.local_random.seed(seed)

        if not os.path.exists('%s/planner/generated_problems' % self.dname):
            os.makedirs('%s/planner/generated_problems' % self.dname)

        info, max_num_repeats, remove_prob = self.get_setup_info(info)
        super(PlannedGameState, self).reset(self.scene_num, False, self.scene_seed, max_num_repeats, remove_prob,
                                            scene=scene, objs=objs)
        self.gt_graph.clear()

        points_source = 'layouts/%s-openable.json' % self.scene_name
        with open(points_source, 'r') as f:
            openable_object_to_point = json.load(f)
        self.openable_object_to_point = openable_object_to_point

        return info

    def should_keep_door_open(self):
        next_plan_action_idx = len(constants.data_dict['plan']['high_pddl'])
        if next_plan_action_idx < len(self.plan):
            next_action = self.plan[next_plan_action_idx]
            return not next_action['action'] in {'GotoLocation', 'End'}
        else:
            return False

    def close_recep(self, recep):
        if recep['openable'] and recep['isOpen']:
            # check if door should be left open for next action
            if self.should_keep_door_open():
                return
            super(PlannedGameState, self).close_recep(recep)
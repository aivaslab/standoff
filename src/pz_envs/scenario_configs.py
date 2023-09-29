from itertools import combinations

import numpy as np


def generate_fillers(timesteps_to_fill, fill_len):
    positions = list(range(1, timesteps_to_fill))
    for c in combinations(positions, fill_len - 1):
        split_positions = [0] + list(c) + [timesteps_to_fill]
        solution = [split_positions[i + 1] - split_positions[i] for i in range(fill_len)]
        yield solution


def parameter_generator(space, params={}):
    if space:
        key, value = next(iter(space.items()))
        if callable(value):
            values = value(params)
        else:
            values = value
        for v in values:
            new_params = params.copy()
            new_params[key] = v
            new_space = {k: v for k, v in space.items() if k != key}
            yield from parameter_generator(new_space, new_params)
    else:
        yield params


def count_non_ob_re(data):
    count = sum(1 for item in data if item not in [['ob'], ['re']])
    return count


def add_bait(events, bait_num, bait_size, uninformed_bait, visible_baits, swap_index='e'):
    if bait_num == uninformed_bait or visible_baits == 0:
        events.extend([['ob'], ['b', bait_size, swap_index], ['re']])
        return len(events) - 2
    else:
        events.append(['b', bait_size, swap_index])
        return len(events) - 1


def add_swap(events, swap_num, swap_indices, uninformed_swap, visible_swaps):
    swap_index, swap_location = swap_indices
    if swap_num == uninformed_swap or visible_swaps == 0:
        events.extend([['ob'], ['sw', swap_index, swap_location], ['re']])
        return len(events) - 2
    else:
        events.append(['sw', swap_index, swap_location])
        return len(events) - 1


def remove_unnecessary_sequences(events):
    removed_indices = []
    for i in range(len(events) - 1, 0, -1):
        if events[i] == ['ob'] and events[i - 1] == ['re']:
            removed_indices.extend([i - 1, i])
            del events[i - 1:i + 1]

    if events and events[-1] == ['re']:
        events.pop()

    # deal with swap indices being removed before swaps or delayed baits
    for event in events:
        if event[0] == 'sw' or event[0] == 'b':
            count = sum(1 for i in removed_indices if i < event[1])
            if event[0] != 'b':
                # event[1] for baits is the size, not an index/location
                event[1] -= count
            if event[2] != 'e':
                count = sum(1 for i in removed_indices if i < event[2])
                event[2] -= count

    return events


def count_permutations(event_list):
    permutations = []
    num_locations = 5
    for event in event_list:
        if len(event) == 3 and event[2] == 'e':
            if event[0] == 'b':
                permutations.append(num_locations)
                num_locations -= 1
            elif event[0] == 'sw':
                permutations.append(num_locations)
        else:
            if event[0] == 'b':  # baits to specific locations still remove empty buckets
                num_locations -= 1
            permutations.append(1)
    return permutations


def identify_counterfactuals(events, fsb=False):
    knowledge = {'eb': False, 'es': False, 'lb': False, 'ls': False, 'cflb': False, 'cfls': False}
    vision = True
    treat_sizes = [-1 for _ in range(len(events))]

    fsb_2nd_swap = False

    for k, event in enumerate(events):
        if event[0] == 'ob':
            vision = False
        elif event[0] == 're':
            vision = True
        elif event[0] == 'b':
            if event[1] == 1:
                treat_sizes[k] = 1
                if vision:
                    knowledge['eb'] = True
                    knowledge['lb'] = True
            else:
                treat_sizes[k] = 0
                if vision:
                    knowledge['es'] = True
                    knowledge['ls'] = True
        elif event[0] == 'sw':
            if vision:
                size = treat_sizes[event[1]] if not fsb_2nd_swap else treat_sizes[firstswap]
                if size == 1:
                    knowledge['eb'] = True
                    knowledge['lb'] = True
                    knowledge['cflb'] = False
                    treat_sizes[k] = 1 #this event's size
                elif size == 0:
                    knowledge['es'] = True
                    knowledge['ls'] = True
                    knowledge['cfls'] = False
                    treat_sizes[k] = 0 #this event's size
                if event[2] != 'e': #if fsb or 2st1l event
                    size2 = treat_sizes[event[2]] if not fsb_2nd_swap else treat_sizes[event[1]]
                    if size2 == 1:
                        knowledge['eb'] = True
                        knowledge['lb'] = True
                        knowledge['cflb'] = False
                    elif size2 == 0:
                        knowledge['es'] = True
                        knowledge['ls'] = True
                        knowledge['cfls'] = False
            else:
                size = treat_sizes[event[1]] if not fsb_2nd_swap else treat_sizes[firstswap]
                if size == 1:
                    if knowledge['lb']:
                        knowledge['cflb'] = True
                    knowledge['lb'] = False
                    treat_sizes[k] = 1
                elif size == 0:
                    if knowledge['ls']:
                        knowledge['cfls'] = True
                    knowledge['ls'] = False
                    treat_sizes[k] = 0
                if event[2] != 'e':
                    size2 = treat_sizes[event[2]] if not fsb_2nd_swap else treat_sizes[event[1]]
                    if size2 == 1:
                        if knowledge['lb']:
                            knowledge['cflb'] = True
                        knowledge['lb'] = False
                    elif size2 == 0:
                        if knowledge['ls']:
                            knowledge['cfls'] = True
                        knowledge['ls'] = False
            if fsb:
                fsb_2nd_swap = True
                firstswap = event[1]

    return knowledge['eb'], knowledge['es'], knowledge['lb'], knowledge['ls']#, knowledge['cflb'], knowledge['cfls']


def count_knowledge_combinations(event_lists, knowledges):
    counter = {}
    def tuple_to_key(knowledge_tuple):
        eb, es, lb, ls = knowledge_tuple
        if eb and not lb:
            b_letter = 'F'
        elif eb and lb:
            b_letter = 'T'
        else:
            b_letter = 'N'

        if es and not ls:
            s_letter = 'f'
        elif es and ls:
            s_letter = 't'
        else:
            s_letter = 'n'

        return f'{b_letter}{s_letter}'
        #mapping = ['eb', 'es', 'lb', 'ls']
        #return '-'.join([mapping[i] for i, val in enumerate(knowledge_tuple) if val])

    name_from_knowledge = {}
    for name in event_lists:
        key = tuple_to_key(knowledges[name])
        counter[key] = counter.get(key, 0) + 1
        if key in name_from_knowledge:
            name_from_knowledge[key].update({name: event_lists[name]})
        else:
            name_from_knowledge[key] = {name: event_lists[name]}

    return counter, name_from_knowledge

class ScenarioConfigs:
    def __init__(self):
        self.parameter_space = {
            # baits and swaps main
            "visible_baits": [0, 1, 2],  # x
            "swaps": [0, 1, 2],  # y
            "visible_swaps": lambda params: range(params['swaps'] + 1),  # z

            # baits and swaps special
            "first_swap_is_both": lambda params: [True, False] if params['swaps'] > 0 else [False],
            "delay_2nd_bait": lambda params: [True, False] if params['swaps'] > 0 and params[
                'first_swap_is_both'] is False else [False],
            "second_swap_to_first_loc": lambda params: [True, False] if params['swaps'] == 2 and params[
                'delay_2nd_bait'] is False else [False],

            # opponent and preferences
            "num_opponents": [0, 1],
            # 2+ requires rewrite of most other things, and more parameters
            "subject_is_dominant": lambda params: [0, 1] if params["num_opponents"] > 0 else [0],
            # a dominant subject is just as trivial as 0 opponents. There is no need for both.
            "opponent_prefers_small": lambda params: [0, 1] if params["num_opponents"] > 0 else [0],
            # we do not also use subject_prefers_small because small here just means different pref, not meaningful size
            "encourage_sharing": lambda params: [0, 1] if params["num_opponents"] > 0 else [0],

            # baits and swaps ordering (randomized during all training)
            "first_bait_size": [0, 1],  # i
            "first_swap_index": lambda params: [0, 1] if params['swaps'] > 0 and params['delay_2nd_bait'] is False and
                                                         params['first_swap_is_both'] is False else [0],
            "uninformed_bait": lambda params: [0, 1] if params['visible_baits'] == 1 else [-1],
            "uninformed_swap": lambda params: [0, 1] if params['swaps'] == 2 and params['visible_swaps'] == 1 else [-1],
        }

        self.all_event_lists = {}
        self.informed_event_lists = {}
        self.uninformed_event_lists = {}
        self.event_list_knowledge = {}  # name: knowledge
        for params in parameter_generator(self.parameter_space):

            visible_baits = params['visible_baits']
            swaps = params['swaps']
            visible_swaps = params['visible_swaps']

            first_swap_is_both = params['first_swap_is_both']
            second_swap_to_first_loc = params['second_swap_to_first_loc']
            delay_2nd_bait = params['delay_2nd_bait']

            '''opponent_prefers_small = params['opponent_prefers_small']
            encourage_sharing = params['encourage_sharing']
            subject_is_dominant = params['subject_is_dominant']
            num_opponents = params['num_opponents']'''

            first_bait_size = params['first_bait_size']
            uninformed_bait = params['uninformed_bait']
            first_swap = params['first_swap_index']
            uninformed_swap = params['uninformed_swap']

            name = f"b{visible_baits}w{swaps}v{visible_swaps}"
            name += "f" if first_swap_is_both else ""
            name += "s" if second_swap_to_first_loc else ""
            name += "d" if delay_2nd_bait else ""
            # name += "u" if subject_is_dominant else ""
            # name += "p" if opponent_prefers_small else ""
            # name += "h" if encourage_sharing else ""
            name += "-" + str(
                1 * first_bait_size + 2 * (uninformed_bait > 0) + 4 * (uninformed_swap > 0) + 8 * (first_swap > 0))
            events = []

            bait_index = []
            for bait_num in range(1 if delay_2nd_bait else 2):
                bait_size = first_bait_size if bait_num == 0 else 1 - first_bait_size
                index = add_bait(events, bait_num, bait_size, uninformed_bait, visible_baits)
                bait_index.append(index)

            first_swap_index = None
            for swap_num in range(1 if delay_2nd_bait else swaps):
                if swap_num == 0 and first_swap_is_both:
                    first_swap_index = add_swap(events, swap_num, (bait_index[0], bait_index[1]), uninformed_swap,
                                                visible_swaps)
                else:
                    swap_index = bait_index[first_swap] if swap_num == 0 else bait_index[1 - first_swap]
                    swap_location = first_swap_index if (swap_num == 1 and second_swap_to_first_loc) else 'e'
                    first_swap_index = add_swap(events, swap_num, (swap_index, swap_location), uninformed_swap,
                                                visible_swaps)


            if delay_2nd_bait:
                bait_size = 1 - first_bait_size
                index = add_bait(events, 1, bait_size, uninformed_bait, visible_baits, first_swap_index)
                bait_index.append(index)

            for swap_num in range(1 if delay_2nd_bait else 2, swaps):
                swap_index = bait_index[first_swap] if swap_num == 0 else bait_index[1 - first_swap]
                swap_location = first_swap_index if swap_num == 1 and second_swap_to_first_loc else 'e'
                first_swap_index = add_swap(events, swap_num, (swap_index, swap_location), uninformed_swap, visible_swaps)

            events = remove_unnecessary_sequences(events)
            self.event_list_knowledge[name] = identify_counterfactuals(events, fsb=first_swap_is_both)

            self.all_event_lists[name] = events
            if visible_baits == 2 and visible_swaps == swaps:
                self.informed_event_lists[name] = events
            if visible_baits == 0 and visible_swaps == 0:
                self.uninformed_event_lists[name] = events
            # print(name, events)

        counter, self.name_from_knowledge = count_knowledge_combinations(self.all_event_lists, self.event_list_knowledge)
        print(counter)

        #print('total lists', len(all_event_lists), 'informed lists', len(informed_event_lists), 'uninformed lists', len(uninformed_event_lists))

        self.all_event_delays = {}
        total_fillers = 0

        for name, listy in self.all_event_lists.items():
            non_ob = count_non_ob_re(listy)
            num_to_fill = 4
            #fillers = list(generate_fillers(8 - (len(listy) - non_ob), non_ob)) # this subtracts non_ob because each contributed 1 to fill
            fillers = list(generate_fillers(num_to_fill, non_ob))
            if not len(fillers) > 0:
                print(name, listy, non_ob, num_to_fill)
                assert False
            #fillers = [[1] * len(listy)]
            self.all_event_delays[name] = fillers
            total_fillers += len(fillers)

        self.all_event_permutations = {}
        total_products = 0
        for event_name in self.all_event_lists:
            self.all_event_permutations[event_name] = count_permutations(self.all_event_lists[event_name])
            product = np.product(self.all_event_permutations[event_name])
            total_products += product * len(self.all_event_delays[event_name])
        print('list_events', len(self.all_event_lists.items()), 'total fillers', total_fillers, 'total permutations', total_products)

        # generate 'stages' for train and test, where a stage is a list of event lists and parameters
        stage_templates = {
            '0': {'params': 'no-op'},
            '1': {'params': 'defaults'}
        }
        self.stages = {}
        for knowledge_key in self.name_from_knowledge.keys():
            for stage_key, stage_info in stage_templates.items():
                new_key = 'sl-' + knowledge_key + stage_key
                self.stages[new_key] = {'events': self.name_from_knowledge[knowledge_key], **stage_info}

        '''lack_to_generalized = {
            "moved": "0.2.2",
            "partiallyUninformed": "1.0.0",
            "replaced": "1.1.0d",
            "informedControl": "2.0.0",
            "removedUninformed": "2.1.0",
            "swapped": "2.1.0a",
            "removedInformed": "2.1.1",
            "misinformed": "2.2.0c",
            "misinformed2": "2.2.0ac",
        }
    
        matching_names = {}
        for key, value in lack_to_generalized.items():
            matching_names[key] = []
            for name in all_event_lists.keys():
                if name.startswith(value + "-"):
                    matching_names[key].append(name)
    
        for key, value in matching_names.items():
            print(f"For {key}, found matches: {value}")'''

        self.standoff = {
            "defaults": {
                "deterministic": True,
                "hidden": True,
                "share_rewards": False,
                "boxes": 5,
                "sub_valence": 1,
                "dom_valence": 1,
                "subject_is_dominant": False,
                "num_puppets": 1,
                "num_agents": 1,
                "events": [[['bait', 'empty'], ['bait', 'empty']]]
            },
            "no-op": {
                "deterministic": True,
                "hidden": True,
                "share_rewards": False,
                "boxes": 5,
                "sub_valence": 1,
                "dom_valence": 1,
                "subject_is_dominant": False,
                "num_puppets": 0,
                "num_agents": 1,
                "events": [[['bait', 'empty'], ['bait', 'empty']]]
            },
        }

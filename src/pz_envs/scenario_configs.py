from itertools import product


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

    # deal with swap indices being removed before swaps
    for event in events:
        if event[0] == 'sw':
            count = sum(1 for i in removed_indices if i < event[1])
            event[1] -= count
            if event[2] != 'e':
                count = sum(1 for i in removed_indices if i < event[2])
                event[2] -= count

    return events


class ScenarioConfigs:

    parameter_space = {
        # baits and swaps main
        "visible_baits": [0, 1, 2],  # x
        "swaps": [0, 1, 2],  # y
        "visible_swaps": lambda params: range(params['swaps'] + 1),  # z

        # baits and swaps special
        "first_swap_is_both": lambda params: [True, False] if params['swaps'] > 0 else [False],  # ab
        "delay_2nd_bait": lambda params: [True, False] if params['swaps'] > 0 and params['first_swap_is_both'] is False else [False],  # d
        "second_swap_to_first_loc": lambda params: [True, False] if params['swaps'] == 2 and params['delay_2nd_bait'] is False else [False],  # c

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
        "first_swap": lambda params: [0, 1] if params['swaps'] > 0 and params['delay_2nd_bait'] is False and params['first_swap_is_both'] is False else [0],  # j
        "uninformed_bait": lambda params: [0, 1] if params['visible_baits'] == 1 else [-1],  # k
        "uninformed_swap": lambda params: [0, 1] if params['swaps'] == 2 and params['visible_swaps'] == 1 else [-1],  # l
    }

    all_event_lists = {}
    informed_event_lists = {}
    uninformed_event_lists = {}
    for params in parameter_generator(parameter_space):

        visible_baits = params['visible_baits']
        swaps = params['swaps']
        visible_swaps = params['visible_swaps']

        first_swap_is_both = params['first_swap_is_both']
        second_swap_to_first_loc = params['second_swap_to_first_loc']
        delay_2nd_bait = params['delay_2nd_bait']

        opponent_prefers_small = params['opponent_prefers_small']
        encourage_sharing = params['encourage_sharing']
        subject_is_dominant = params['subject_is_dominant']
        num_opponents = params['num_opponents']

        first_bait_size = params['first_bait_size']
        uninformed_bait = params['uninformed_bait']
        first_swap = params['first_swap']
        uninformed_swap = params['uninformed_swap']


        name = f"b{visible_baits}s{swaps}v{visible_swaps}"
        name += "f" if first_swap_is_both else ""
        name += "s" if second_swap_to_first_loc else ""
        name += "d" if delay_2nd_bait else ""
        #name += "u" if subject_is_dominant else ""
        #name += "p" if opponent_prefers_small else ""
        #name += "h" if encourage_sharing else ""
        name += "-" + str(1*first_bait_size + 2*(uninformed_bait > 0) + 4*(uninformed_swap > 0) + 8*(first_swap > 0))
        events = []

        bait_index = []
        for bait_num in range(1 if delay_2nd_bait else 2):
            bait_size = first_bait_size if bait_num == 0 else 1 - first_bait_size
            index = add_bait(events, bait_num, bait_size, uninformed_bait, visible_baits)
            bait_index.append(index)

        first_swap_index = None
        for swap_num in range(1 if delay_2nd_bait else swaps):
            if swap_num == 0 and first_swap_is_both:
                first_swap_index = add_swap(events, swap_num, (bait_index[0], bait_index[1]), uninformed_swap, visible_swaps)
            else:
                swap_index = bait_index[first_swap] if swap_num == 0 else bait_index[1 - first_swap]
                swap_location = first_swap_index if swap_num == 1 and second_swap_to_first_loc else 'e'
                first_swap_index = add_swap(events, swap_num, (swap_index, swap_location), uninformed_swap, visible_swaps)

        if delay_2nd_bait:
            bait_size = 1 - first_bait_size
            index = add_bait(events, 1, bait_size, uninformed_bait, visible_baits, first_swap_index)
            bait_index.append(index)

        for swap_num in range(1 if delay_2nd_bait else 2, swaps):
            swap_index = bait_index[first_swap] if swap_num == 0 else bait_index[1 - first_swap]
            swap_location = first_swap_index if swap_num == 1 and second_swap_to_first_loc else 'e'
            first_swap_index = add_swap(events, swap_num, (swap_index, swap_location), uninformed_swap, visible_swaps)

        events = remove_unnecessary_sequences(events)
        all_event_lists[name] = events
        if visible_baits == 2 and visible_swaps == swaps:
            informed_event_lists[name] = events
        if visible_baits == 0 and visible_swaps == 0:
            uninformed_event_lists[name] = events
        #print(name, events)


    lack_to_generalized = {
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
        print(f"For {key}, found matches: {value}")

    print('total lists', len(all_event_lists), 'informed lists', len(informed_event_lists), 'uninformed lists', len(uninformed_event_lists))

    standoff = {
        "defaults": {
            "deterministic": False,
            "hidden": True,
            "share_rewards": False,
            "boxes": 5,
            "sub_valence": 1,
            "dom_valence": 1,
            "subject_is_dominant": False,  # subordinate has delayed release. for subordinate first, use negative
            "num_puppets": 1,
            "num_agents": 1,
            "events": [[['bait', 'empty'], ['bait', 'empty']]]  # list, event, args
        },
        "random": {
            "num_puppets": [1],
            "events": [[['random']] * n for n in range(2, 6)]
        },
        "stage_1": {
            "num_puppets": [0],
            "boxes": [5],
        },
        "stage_2": {
            "num_puppets": [1],
            "boxes": [5],
        },
        "s2b": {
            "num_puppets": [1, 0],
        },
        "informedControl": {
            "events": [[['bait', 'empty'], ['bait', 'empty']]]  # optimal: Small
        },
        "partiallyUninformed": {  # optimal: if big bait obscured, Big, else Small
            "events": [[['bait', 'empty'], ['obscure'], ['bait', 'empty']],
                       [['obscure'], ['bait', 'empty'], ['reveal'], ['bait', 'empty']]]
        },
        "removedInformed1": {  # optimal: Neither (minor reward preference over copying dominant's decision)
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['remove', x]] for x in [0, 1]]
        },
        "removedUninformed1": {  # optimal: if Big is removed, Small, else Neither
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', x]] for x in [0, 1]]
        },
        "removedInformed2": {  # optimal: small
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['swap', x, 'empty']] for x in [0, 1]]
        },
        "removedUninformed2": {  # optimal: if Big is swapped, big, else small
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', x, 'empty']] for x in [0, 1]]
        },
        "moved": {  # optimal: Small
            "events": [[['obscure'], ['bait', 'empty'], ['bait', 'empty'], ['reveal'], ['swap', 1, 'empty'],
                        ['swap', 2, 'empty']]]
        },
        "replaced": {  # optimal: if first bait is big, Big, else Small
            "events": [[['bait', 'empty'], ['obscure'], ['swap', 0, 'empty'], ['bait', 0]]]
        },
        "misinformed": {
            # optimal: if big is swapped with empty, Big. If big swapped with small, also big. If small swapped with empty, small. If small swapped with big, big.
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', x, 'else'], ['swap', x, 'empty']]
                       for x in [0, 1]]
        },
        "swapped": {  # optimal: Big
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 1]]]
        },
        "all": {
            "events": [[['bait', 'empty'], ['bait', 'empty']],
                       [['bait', 'empty'], ['obscure'], ['bait', 'empty']],
                       [['obscure'], ['bait', 'empty'], ['reveal'], ['bait', 'empty']],
                       # [['bait', 'empty'], ['bait', 'empty'], ['remove', 0]],
                       # [['bait', 'empty'], ['bait', 'empty'], ['remove', 1]],
                       [['bait', 'empty'], ['bait', 'empty'], ['swap', 0, 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['swap', 1, 'empty']],

                       # [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', 0]],
                       # [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', 1]],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 1, 'empty']],
                       [['obscure'], ['bait', 'empty'], ['bait', 'empty'], ['reveal'], ['swap', 1, 'empty'],
                        ['swap', 2, 'empty']],
                       [['bait', 'empty'], ['obscure'], ['swap', 0, 'empty'], ['bait', 0]],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 'else'], ['swap', 0, 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 1, 'else'], ['swap', 1, 'empty']],
                       [['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 1]]]
        }
    }
    standoff_optimal_policies = {
        "stage_1": 'big',
        "swapped": 'big',
        "partiallyUninformed": 'b-s',
        "replaced": 'b-s',
        "misinformed": 'b-s',
        "stage_2": 'small',
        "informedControl": 'small',
        "moved": 'small',
        "removedUninformed": 's-n',
        "removedInformed": 'none',
    }

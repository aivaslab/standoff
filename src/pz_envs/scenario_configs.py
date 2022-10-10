class ScenarioConfigs:
    tutorial = {
        "empty": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "a"
        },
        "empty_clutter": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "b"
        },
        "grid": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "c"
        },
        "grid_clutter": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "d"
        },
        "empty_hide": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "e"
        },
        "empty_hide_redherring": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "f"
        },
        "empty_hide_preference": {
            "num_puppets": 0,
            "eType": "t",
            "eVar": "g"
        },
        # 2 eVariants are all doorkey
        # 3 eVariants are also doorkey
        "nav_no_goal": {
            "num_puppets": 0,
            "eType": "n",
            "eVar": "a"
        },
        "nav_simple": {
            "num_puppets": 0,
            "eType": "n",
            "eVar": "b"
        },
        # ...
    }

    standoff = {
        "defaults": {
            "hidden": True,
            "sharedRewards": False,
            "boxes": 5,
            "sub_valence": 1,
            "dom_valence": 1,
            "subject_is_dominant": False,  # subordinate has delayed release. for subordinate first, use negative
            "lavaHeight": 3,  # should be odd, >=2
            "num_puppets": 1,
            "num_agents": 1,
            "events": [[['bait', 'empty'], ['bait', 'empty']]] #list, event, args
        },

        "tutorial_step_1": {
            "num_puppets": [0],
            "boxes": [5],
            "hidden": [False, True],
        },
        "tutorial_step_1a": {
            "num_puppets": [0],
            "boxes": [5],
            "hidden": [False],
        },
        "tutorial_step_1b": {
            "num_puppets": [0],
            "boxes": [5],
            "hidden": [True],
        },
        "tutorial_step_1c": {
            "num_puppets": [1],
            "boxes": [5],
            "hidden": [False],
        },
        "tutorial_step_2": {
            # more eVaried training, including easier cases than eval
            "num_puppets": [1],
            "boxes": [5],
            "hidden": [True],
        },
        "informed control": {
            "events": [[['bait', 'empty'], ['bait', 'empty']]]
        },
        "partially uninformed": {
            "events": [[['bait', 'empty'], ['obscure'], ['bait', 'empty']],
                       [['obscure'], ['bait', 'empty'], ['reveal'], ['bait', 'empty']]]
        },
        "removed informed": {
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['remove', x]] for x in [0, 1]]
        },
        "removed uninformed": {
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['remove', x]] for x in [0, 1]]
        },
        "moved": {
            "events": [[['obscure'], ['bait', 'empty'], ['bait', 'empty'], ['reveal'], ['swap', 1, 'empty'],
                       ['swap', 2, 'empty']]]
        },
        "replaced": {
            "events": [[['bait', 'empty'], ['obscure'], ['swap', 0, 'empty'], ['bait', 0]]]
        },
        "misinformed": {
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', x, 'else']] for x in [0, 1]]
        },
        "swapped": {
            "events": [[['bait', 'empty'], ['bait', 'empty'], ['obscure'], ['swap', 0, 1]]]
        }
    }


class AllParams():
    params = {"standoffEnv": {
        "adversarial": [True, False],
        "hidden": [True, False],
        "rational": [True, False],
        "sharedRewards": [True, False],
        "firstBig": [True, False],  # whether we place big first
        "boxes": [2, 3, 4, 5],
        "num_puppets": [0, 1, 2],
        "followDistance": [0, 1],  # 0 = d first, 1=sub first
        "lavaHeight": [2],
        "baits": [1, 2],
        "baitSize": [1, 2],
        "informed": ['informed', 'uninformed', 'fake', 'half1', 'half2'],
        "swapType": ['swap', 'replace', 'remove', 'move', 'mis'],
        "visibility": ['full', 'curtains'],  # keys, invisibility potion
        "cause": ['blocks', 'direction', 'accident', 'inability'],
        "lava": ['lava', 'block'],
        "num_agents": [1, 2],
        "num_puppets": [0, 1, 2]
    }
    }

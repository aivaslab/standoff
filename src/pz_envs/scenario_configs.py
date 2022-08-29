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
            "adversarial": True,
            "hidden": True,
            "rational": True,
            "sharedRewards": False,
            "firstBig": True,
            "boxes": 5,
            "sub_valence": 1,
            "dom_valence": 1,
            "followDistance": 1,  # subordinate has delayed release. for subordinate first, use negative
            "lavaHeight": 3,  # should be odd, >=2
            "baits": 1,
            "baitSize": 2,
            "informed": 'informed',
            "swapType": 'swap',
            "visibility": 'curtains',  # keys, invisibility potion
            "cause": 'blocks',
            "lava": 'lava',
            "num_puppets": 1,
            "num_agents": 1,
        },

        "tutorial_step_1": {
            "num_puppets": [0],
            "boxes": [5],
            "baitSize": [1, 2],
            "followDistance": [0, 1],
            "visibility": ['full', 'curtains'],
            "informed": "informed",
            "hidden": [False, True],
            "events": [('bait', 'empty'), ('bait', 'empty')]
        },
        "tutorial_step_1a": {
            "num_puppets": [0],
            "boxes": [5],
            "baitSize": [1, 2],
            "followDistance": [0, 1],
            "visibility": ['full'],
            "informed": "informed",
            "hidden": [False],
            "events": [('bait', 'empty'), ('bait', 'empty')]
        },
        "tutorial_step_1b": {
            "num_puppets": [0],
            "boxes": [5],
            "baitSize": [1, 2],
            "followDistance": [0, 1],
            "visibility": ['full'],
            "informed": "informed",
            "hidden": [True],
            "events": [('bait', 'empty'), ('bait', 'empty')]
        },
        "tutorial_step_2": {
            # more eVaried training, including easier cases than eval
            "boxes": [5],
            "baitSize": [2],
            "followDistance": [0, 1],
            "visibility": ['curtains'],
            "informed": "informed",
            "hidden": [False, True],
            "events": [('bait', 'empty'), ('bait', 'empty')]
        },
        "tutorial_step_2a": {
            # single test config that must be passed
            "boxes": [5],
            "baitSize": [2],
            "visibility": 'curtains',
            "informed": "informed",
            "hidden": False,
            "followDistance": [1],
            "events": [('bait', 'empty'), ('bait', 'empty')]
        },
        "tutorial_step_2b": {
            # single test config that must be passed
            "boxes": [5],
            "baitSize": [2],
            "visibility": 'curtains',
            "informed": "informed",
            "hidden": True,
            "followDistance": [1],
            "events": [('bait', 'empty'), ('bait', 'empty')]
        },
        "informed control": {
            "informed": 'informed',
            "events": [('bait', 'empty'), ('bait', 'empty')]
        },
        "partially uninformed": {
            "informed": ['half1', 'half2'],
            "firstBig": [True, False],
            "baitSize": 1,
            "baits": 2,
            "events": [[('bait', 'empty'), ('obscure'), ('bait', 'empty')],
                       [('obscure'), ('bait', 'empty'), ('reveal'), ('bait', 'empty')]]
        },
        "removed informed": {
            "informed": "informed",
            "swapType": 'remove',
            "baitSize": 2,
            "baits": 3,
            "events": [[('bait', 'empty'), ('bait', 'empty'), ('remove', x)] for x in [0, 1]]
        },
        "removed uninformed": {
            "informed": "uninformed",
            "swapType": 'remove',
            "baitSize": 2,
            "baits": 2,
            "events": [[('bait', 'empty'), ('bait', 'empty'), ('block'), ('remove', x)] for x in [0, 1]]
        },
        "moved": {
            "informed": "informed",  # but uninformed about first baiting
            "swapType": 'move',
            "baitSize": 2,
            "baits": 2,
            "events": [('obscure'), ('bait', 'empty'), ('bait', 'empty'), ('reveal'), ('move', 0, 'empty'),
                       ('move', 1, 'empty')]
        },
        "replaced": {
            "informed": "uninformed",
            "swapType": 'replace',
            "baitSize": 1,
            "baits": 3,
            "events": [('bait', 'empty'), ('obscure'), ('move', 0, 'empty'), ('bait', 0)]
        },
        "misinformed": {
            "informed": "uninformed",
            "swapType": ['swap', 'replace'],  # any bucket swapped with a food
            "baitSize": 2,
            "baits": 2,
            "events": [[('bait', 'empty'), ('bait', 'empty'), ('obscure'), ('move', x, 'anyElse')] for x in [0, 1]]
        },
        "swapped": {
            "informed": "uninformed",
            "swapType": 'swap',
            "baitSize": 2,
            "baits": 2,
            "events": [('bait', 'empty'), ('bait', 'empty'), ('obscure'), ('move', 0, 1)]
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

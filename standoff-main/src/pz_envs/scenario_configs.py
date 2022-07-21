


class ScenarioConfigs():
    tutorial = {
        "empty": {
            "puppets": 0,
            "eType": "t",
            "eVar": "a"
        },
        "empty_clutter": {
            "puppets": 0,
            "eType": "t",
            "eVar": "b"
        },
        "grid": {
            "puppets": 0,
            "eType": "t",
            "eVar": "c"
        },
        "grid_clutter": {
            "puppets": 0,
            "eType": "t",
            "eVar": "d"
        },
        "empty_hide": {
            "puppets": 0,
            "eType": "t",
            "eVar": "e"
        },
        "empty_hide_redherring": {
            "puppets": 0,
            "eType": "t",
            "eVar": "f"
        },
        "empty_hide_preference": {
            "puppets": 0,
            "eType": "t",
            "eVar": "g"
        },
        #2 eVariants are all doorkey
        #3 eVariants are also doorkey
        "nav_no_goal": {
            "puppets": 0,
            "eType": "n",
            "eVar": "a"
        },
        "nav_simple": {
            "puppets": 0,
            "eType": "n",
            "eVar": "b"
        },
        #...
    }

    standoff = {
        "tutorial_step_1": {
            "puppets": 0,
            "boxes": [5],
            "baitSize": [1,2],
            "followDistance": [0,1],
            "visibility": ['full', 'curtains'],
            "informed": "informed",
            "hidden": [False, True]
        },
        "tutorial_step_1a": {
            "puppets": 0,
            "boxes": [5],
            "baitSize": [1,2],
            "followDistance": [0,1],
            "visibility": ['full'],
            "informed": "informed",
            "hidden": [False]
        },
        "tutorial_step_1b": {
            "puppets": 0,
            "boxes": [5],
            "baitSize": [1,2],
            "followDistance": [0,1],
            "visibility": ['full'],
            "informed": "informed",
            "hidden": [True]
        },
        "tutorial_step_2": {
            #more eVaried training, including easier cases than eval
            "puppets": 1,
            "boxes": [5],
            "baitSize": [2],
            "followDistance": [0,1],
            "visibility": ['curtains'],
            "informed": "informed",
            "hidden": [False, True]
        },
        "tutorial_step_2_eval": {
            #single test config that must be passed
            "puppets": 1,
            "boxes": [5],
            "baitSize": [2],
            "visibility": 'curtains',
            "informed": "informed",
            "hidden": True
        },
        "informed control": {
            "informed": 'informed',
        },
        "partially uninformed": {
            "informed": ['half1', 'half2'],
            "firstBig": [True, False],
            "baitSize": 1,
            "baits": 2,
        },    
        "removed informed": {
            "informed": "informed",
            "swapType": 'remove',
            "baitSize": 2,
            "baits": 3,
        },
        "removed uninformed": {
            "informed": "uninformed",
            "swapType": 'remove',
            "baitSize": 2,
            "baits": 2,
        },
        "moved": {
            "informed": "informed", #but uninformed about first baiting
            "swapType": 'move',
            "baitSize": 2,
            "baits": 2,
        },
        "replaced": {
            "informed": "uninformed",
            "swapType": 'replace',
            "baitSize": 1,
            "baits": 3,
        },
        "misinformed": {
            "informed": "uninformed",
            "swapType": ['swap', 'replace'], #any bucket swapped with a food
            "baitSize": 2,
            "baits": 2,
        },
        "swapped": {
            "informed": "uninformed",
            "swapType": 'swap',
            "baitSize": 2,
            "baits": 2,
        }
    }


class AllParams():

    params = {"standoffEnv":  {
            "adversarial": [True, False],
            "hidden": [True, False],
            "rational": [True, False],
            "sharedRewards": [True, False],
            "firstBig": [True, False],#whether we place big first
            "boxes": [2,3,4,5],
            "puppets": [0,1,2],
            "followDistance": [0,1], #0 = d first, 1=sub first
            "lavaHeight": [2],
            "baits": [1,2],
            "baitSize": [1,2],
            "informed": ['informed', 'uninformed', 'fake', 'half1', 'half2'],
            "swapType": ['swap', 'replace', 'remove', 'move', 'mis'],
            "visibility": ['full', 'curtains'], #keys, invisibility potion
            "cause": ['blocks', 'direction', 'accident', 'inability'],
            "lava": ['lava', 'block'],
            }
        }

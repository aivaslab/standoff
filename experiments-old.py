if 159 in todo:
    print('Running experiment 159: homogeneous')

    combined_path_list = []
    last_path_list = []
    lp_list = []
    key_param = 'regime'
    key_param_list = []
    session_params['oracle_is_target'] = False

    for regime in list(hregime.keys()):
        print('regime:', regime, 'train_sets:', hregime[regime])
        combined_paths, last_epoch_paths, lp = run_supervised_session(
            save_path=os.path.join('supervised', exp_name, regime),
            train_sets=hregime[regime],
            eval_sets=regimes['everything'],
            oracle_labels=[None],
            key_param=key_param,
            key_param_value=regime,
            label='correct-box',
            **session_params
        )
        last_path_list.append(last_epoch_paths)
        combined_path_list.append(combined_paths)
        key_param_list.append(regime)
        lp_list.append(lp)

    do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

if 259 in todo:
    print('Running experiment 259: homogeneous but box location')

    combined_path_list = []
    last_path_list = []
    lp_list = []
    key_param = 'regime'
    key_param_list = []
    session_params['oracle_is_target'] = False

    for regime in list(hregime.keys()):
        print('regime:', regime, 'train_sets:', hregime[regime])
        combined_paths, last_epoch_paths, lp = run_supervised_session(
            save_path=os.path.join('supervised', exp_name, regime),
            train_sets=hregime[regime],
            eval_sets=regimes['everything'],
            oracle_labels=[None],
            key_param=key_param,
            key_param_value=regime,
            label='shouldGetBig',
            **session_params
        )
        last_path_list.append(last_epoch_paths)
        combined_path_list.append(combined_paths)
        key_param_list.append(regime)
        lp_list.append(lp)

    do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

if 60 in todo:
    print('Running experiment 59: homogeneous with b-loc')

    combined_path_list = []
    last_path_list = []
    lp_list = []
    key_param = 'regime'
    key_param_list = []
    session_params['oracle_is_target'] = True

    for regime in list(hregime.keys()):
        print('regime:', regime, 'train_sets:', hregime[regime])
        combined_paths, last_epoch_paths, lp = run_supervised_session(
            save_path=os.path.join('supervised', exp_name, regime),
            train_sets=hregime[regime],
            eval_sets=regimes['everything'],
            oracle_labels=['b-loc'],
            key_param=key_param,
            key_param_value=regime,
            **session_params
        )
        last_path_list.append(last_epoch_paths)
        combined_path_list.append(combined_paths)
        key_param_list.append(regime)
        lp_list.append(lp)

    do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

if 61 in todo:
    print('Running experiment 59: homogeneous with b-loc (supplied)')

    combined_path_list = []
    last_path_list = []
    lp_list = []
    key_param = 'regime'
    key_param_list = []
    session_params['oracle_is_target'] = True

    for regime in list(hregime.keys()):
        print('regime:', regime, 'train_sets:', hregime[regime])
        combined_paths, last_epoch_paths, lp = run_supervised_session(
            save_path=os.path.join('supervised', exp_name, regime),
            train_sets=hregime[regime],
            eval_sets=regimes['everything'],
            oracle_labels=['b-loc'],
            key_param=key_param,
            key_param_value=regime,
            **session_params
        )
        last_path_list.append(last_epoch_paths)
        combined_path_list.append(combined_paths)
        key_param_list.append(regime)
        lp_list.append(lp)

    do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

if 62 in todo:
    print('Running experiment 62: special')

    combined_path_list = []
    last_path_list = []
    lp_list = []
    key_param = 'regime'
    key_param_list = []
    session_params['oracle_is_target'] = False

    for regime in list(sregime.keys()):
        print('regime:', regime, 'train_sets:', sregime[regime])
        combined_paths, last_epoch_paths, lp = run_supervised_session(
            save_path=os.path.join('supervised', exp_name, regime),
            train_sets=sregime[regime],
            eval_sets=regimes['everything'],
            oracle_labels=[None],
            key_param=key_param,
            key_param_value=regime,
            **session_params
        )
        last_path_list.append(last_epoch_paths)
        combined_path_list.append(combined_paths)
        key_param_list.append(regime)
        lp_list.append(lp)

    do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

if 63 in todo:
    print('Running experiment 63: homogeneous with varied oracles and oracle types')

    combined_path_list = []
    last_path_list = []
    lp_list = []
    key_param = 'oracle'
    key_param_list = []
    regime = 'homogeneous'

    for oracle_label in ['', 'b-loc', 'loc', 'target-loc', 'target-size', 'b-exist']:
        oracle_types = ['target', 'linear', 'early'] if oracle_label != '' else ['']
        for oracle_type in oracle_types:
            session_params['oracle_is_target'] = True if oracle_type == 'target' else False
            this_name = oracle_label + '_' + oracle_type
            print('regime:', regime, 'train_sets:', hregime[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, this_name),
                train_sets=hregime[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[oracle_label] if oracle_label != '' else [],
                key_param=key_param,
                key_param_value=this_name,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(this_name)

    do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

if 64 in todo:
    print('Running experiment 64: tt with varied oracles and oracle types')

    combined_path_list = []
    last_path_list = []
    lp_list = []
    key_param = 'oracle'
    key_param_list = []
    regime = 'Tt'

    for oracle_label in ['b-loc', 'loc', 'target-loc', 'target-size', 'b-exist']:
        oracle_types = ['linear', 'early'] if oracle_label != '' else ['']
        for oracle_type in oracle_types:
            session_params['oracle_is_target'] = True if oracle_type == 'target' else False
            this_name = oracle_label + '_' + oracle_type
            print('regime:', regime, 'train_sets:', mixed_regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, this_name),
                train_sets=mixed_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[oracle_label] if oracle_label != '' else [],
                key_param=key_param,
                key_param_value=this_name,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(this_name)

    do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)


   if 1 in todo:
        print('Running experiment 1: varied models training directly on the test set')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'

        for regime in list(regimes.keys()):
            print('regime:', regime)
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)

        do_comparison(combined_path_list, last_path_list, regimes, key_param, exp_name, params, prior_metrics, lp_list)

    if 51 in todo:
        print('Running experiment 51: single models do not generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        used_regimes = []

        for regime in list(single_regimes.keys()):
            print('regime:', regime)
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=single_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)
            used_regimes.append(single_regimes[regime])

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, used_regimes)

    if 151 in todo:
        print('Running experiment 151: single models do not generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        used_regimes = []

        for regime in list(single_regimes.keys()):
            print('regime:', regime)
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=single_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                label='correct-box',
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)
            used_regimes.append(single_regimes[regime])

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, used_regimes)\

    if 251 in todo:
        print('Running experiment 251: single models do not generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        used_regimes = []

        for regime in list(single_regimes.keys()):
            print('regime:', regime)
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=single_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                label='shouldGetBig',
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)
            used_regimes.append(single_regimes[regime])

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, used_regimes)

    if 52 in todo:
        print('Running experiment 52: (small version of 51) single models do not generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(single_regimes.keys())[:2]:
            print('regime:', regime, 'train_sets:', single_regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=single_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 53 in todo:
        print('Running experiment 53: multi models do generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(regimes.keys()):
            print('regime:', regime, 'train_sets:', regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 153 in todo:
        print('Running experiment 153: multi models do generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(regimes.keys()):
            print('regime:', regime, 'train_sets:', regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                label='correct-box',
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 253 in todo:
        print('Running experiment 253: multi models do generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(regimes.keys()):
            print('regime:', regime, 'train_sets:', regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                label='shouldGetBig',
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 54 in todo:
        print('Running experiment 54: mixed models maybe generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(mixed_regimes.keys()):
            print('regime:', regime, 'train_sets:', mixed_regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=mixed_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 154 in todo:
        print('Running experiment 154: mixed models maybe generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(mixed_regimes.keys()):
            print('regime:', regime, 'train_sets:', mixed_regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=mixed_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                label='correct-box',
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 254 in todo:
        print('Running experiment 254: mixed models maybe generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(mixed_regimes.keys()):
            print('regime:', regime, 'train_sets:', mixed_regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=mixed_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                label='shouldGetBig',
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 55 in todo:
        print('Running experiment 55: mixed models with oracle training maybe generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = True

        for regime in list(mixed_regimes.keys()):
            print('regime:', regime, 'train_sets:', mixed_regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=mixed_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=['b-loc'],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 85 in todo:
        print('Running experiment 85: mixed models with oracle-supplied training maybe generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = False

        for regime in list(mixed_regimes.keys()):
            print('regime:', regime, 'train_sets:', mixed_regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=mixed_regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=['b-loc'],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)


    if 56 in todo:
        print('Running experiment 56: odd-one-out')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []

        for regime in list(leave_one_out_regimes.keys()):
            for oracle in [0, 1]:
                session_params['oracle_is_target'] = bool(oracle)
                print('regime:', regime, 'train_sets:', leave_one_out_regimes[regime])
                combined_paths, last_epoch_paths, lp = run_supervised_session(
                    save_path=os.path.join('supervised', exp_name, regime + '_' + str(oracle)),
                    train_sets=leave_one_out_regimes[regime],
                    eval_sets=regimes['everything'],
                    oracle_labels=[None if not oracle else 'b-loc'],
                    key_param=key_param,
                    key_param_value=regime,
                    **session_params
                )
                last_path_list.append(last_epoch_paths)
                combined_path_list.append(combined_paths)
                key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 57 in todo:
        print('Running experiment 57: progression')

        key_param = 'regime'
        all_accuracies = {}
        save_file = os.path.join('supervised', exp_name, 'prog.pkl')
        image_file = os.path.join('supervised', exp_name, 'prog.png')

        session_params['skip_calc'] = True
        session_params['skip_activations'] = True
        session_params['repetitions'] = 1

        for progression_trial in range(5):
            base_regimes = ['sl-' + x + '0' for x in sub_regime_keys]
            add_regimes = ['sl-' + x + '1' for x in sub_regime_keys]
            random.shuffle(add_regimes)

            prog_regimes = [[x for x in base_regimes]]
            for idx in range(9):
                prog_regimes.append(copy.copy(prog_regimes[idx]))
                prog_regimes[idx + 1].append(add_regimes.pop())

            for oracle in [0, 1]:
                prog_accuracies = []
                last_path_list = []
                lp_list = []
                key_param_list = []
                session_params['oracle_is_target'] = bool(oracle)
                for regime in range(10):
                    print('regime:', regime, 'train_sets:', prog_regimes[regime])

                    _, last_epoch_paths, lp = run_supervised_session(
                        save_path=os.path.join('supervised', exp_name, str(oracle) + '_' + str(progression_trial) + '_' + str(regime)),
                        train_sets=prog_regimes[regime],
                        eval_sets=regimes['direct'],
                        oracle_labels=[None if not oracle else 'b-loc'],
                        key_param=key_param,
                        key_param_value=str(regime),
                        **session_params
                    )
                    last_path_list.append(last_epoch_paths)
                    key_param_list.append(str(regime))

                    replace_dict = {'1': 1, '0': 0}
                    #print('last path', last_epoch_paths)
                    df_list = []
                    if len(last_epoch_paths):
                        for df_path in last_epoch_paths:
                            chunks = pd.read_csv(df_path, chunksize=10000)
                            for chunk in chunks:
                                chunk.replace(replace_dict, inplace=True)
                                df_list.append(chunk)
                        combined_df = pd.concat(df_list, ignore_index=True)
                        avg_accuracy = combined_df['accuracy'].mean()
                        print('avg_accuracy', avg_accuracy)
                        prog_accuracies.append(avg_accuracy)

                all_accuracies[str(oracle) + '_' + str(progression_trial)] = prog_accuracies # list is length regimes

        with open(save_file, 'wb') as f:
            pickle.dump(all_accuracies, f)

        plot_progression(save_file, image_file)

    if 58 in todo:
        print('Running experiment 58: multi models with oracle training maybe generalize')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = True

        for regime in list(regimes.keys()):
            print('regime:', regime, 'train_sets:', regimes[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=regimes[regime],
                eval_sets=regimes['everything'],
                oracle_labels=['b-loc'],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

    if 59 in todo:
        print('Running experiment 59: homogeneous')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = False

        for regime in list(hregime.keys()):
            print('regime:', regime, 'train_sets:', hregime[regime])
            combined_paths, last_epoch_paths, lp = run_supervised_session(
                save_path=os.path.join('supervised', exp_name, regime),
                train_sets=hregime[regime],
                eval_sets=regimes['everything'],
                oracle_labels=[None],
                key_param=key_param,
                key_param_value=regime,
                **session_params
            )
            last_path_list.append(last_epoch_paths)
            combined_path_list.append(combined_paths)
            key_param_list.append(regime)
            lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)

        if 's' in todo:
            print('making combined heatmap')
            session_params['skip_train'] = True
            session_params['skip_eval'] = True
            session_params['skip_calc'] = True

            key_param = 'regime'
            regime_map = {'exp_54': mixed_regimes, 'exp_51': single_regimes, 'exp_53': regimes, 'exp_59': hregime,
                          'exp_154': mixed_regimes, 'exp_151': single_regimes, 'exp_153': regimes, 'exp_159': hregime,
                          'exp_254': mixed_regimes, 'exp_251': single_regimes, 'exp_253': regimes, 'exp_259': hregime}

            for x in [0, 1, 2]:
                if x in todo:
                    if x > 0:
                        names = [f'exp_{x}53', f'exp_{x}51', f'exp_{x}54', f'exp_{x}59']
                    else:
                        names = ['exp_53', 'exp_51', 'exp_54', 'exp_59']
                    df_path_list = []
                    df_path_list2 = []
                    key_param_list = []
                    for dataset in names:
                        if last_timestep:
                            real_name = dataset + "-L"
                        else:
                            real_name = dataset
                        print(real_name)
                        cur_regimes = regime_map[dataset]
                        for regime in list(cur_regimes.keys()):
                            tset = cur_regimes[regime]
                            all_dfs, last_epoch_paths = run_supervised_session(
                                save_path=os.path.join('supervised', real_name, regime),
                                train_sets=tset,
                                eval_sets=regimes['everything'],
                                oracle_labels=[None],
                                key_param=key_param,
                                key_param_value=regime,
                                **session_params
                            )
                            df_path_list2.append(all_dfs)
                            df_path_list.append(last_epoch_paths)
                            key_param_list.append(regime)

                    # print(key_param_list)
                    path = os.path.join('supervised', f'special-{x}')
                    if not os.path.exists(path):
                        os.mkdir(path)
                    special_heatmap(df_path_list2, df_path_list, 'regime', key_param_list, names, path, params)


    if 101 in todo:
        print('Running experiment 101: base, different models and answers')

        combined_path_list = []
        last_path_list = []
        lp_list = []
        key_param = 'regime'
        key_param_list = []
        session_params['oracle_is_target'] = False

        for label, label_name in [('correct-loc', 'loc')]:
            for model_type in ['smlp', 'cnn', 'clstm', ]:
                for regime in ['everything']:
                    kpname = model_type + '-' + label_name
                    print(model_type + '-' + label_name, 'regime:', regime, 'train_sets:', regimes['everything'])
                    combined_paths, last_epoch_paths, lp = run_supervised_session(
                        save_path=os.path.join('supervised', exp_name, kpname),
                        train_sets=regimes['everything'],
                        eval_sets=regimes['everything'],
                        oracle_labels=[None],
                        key_param=key_param,
                        key_param_value=kpname,
                        label=label,
                        model_type=model_type,
                        **session_params
                    )
                    last_path_list.append(last_epoch_paths)
                    combined_path_list.append(combined_paths)
                    key_param_list.append(kpname)
                    lp_list.append(lp)

        do_comparison(combined_path_list, last_path_list, key_param_list, key_param, exp_name, params, prior_metrics, lp_list)
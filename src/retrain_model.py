

def retrain_model(test_sets, target_label, load_path='supervised/', model_load_path='', oracle_labels=[], repetition=0, save_labels=True,
                  epoch_number=0, prior_metrics=[], last_timestep=True, act_label_names=None, model_type=None, model_kwargs=None,
                  seed=0, test_percent=0.2, retrain_batches=5000, retrain_path=''):
    test_loaders, special_criterion, oracle_criterion, model, device, \
    data, labels, params, oracles, act_labels, batch_size, prior_metrics_data, \
    model_kwargs = load_model_data_eval_retrain(test_sets, load_path,
                                                target_label, last_timestep,
                                                prior_metrics, model_load_path,
                                                repetition,
                                                model_type, oracle_labels,
                                                save_labels, act_label_names,
                                                test_percent)
    print("retraining model on all")
    train_indices = filter_indices(data, labels, params, oracles, is_train=True, test_percent=test_percent)
    test_indices = filter_indices(data, labels, params, oracles, is_train=False, test_percent=test_percent)
    train_dataset = TrainDatasetBig(data, labels, params, oracles, train_indices, metrics=prior_metrics_data)
    val_dataset = TrainDatasetBig(data, labels, params, oracles, test_indices, metrics=prior_metrics_data)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    for optimizer in ['adam']:  # ['sgd', 'adam', 'momentum09', 'nesterov09', 'rmsprop']:
        train_model_retrain(model, train_loader, val_loader, retrain_path, model_kwargs=model_kwargs, opt=optimizer, max_tries=5, repetition=repetition, max_batches=retrain_batches)


def train_model_retrain(model, train_loader, test_loader, save_path,
                        model_kwargs=None,
                        oracle_labels=[],
                        repetition=0,
                        save_models=True,
                        record_loss=True,
                        oracle_is_target=False,
                        max_batches=10000,
                        tolerance=5e-4,
                        max_tries=3,
                        opt='sgd'):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    lr = model_kwargs['lr']
    old_weights = [param.data.clone() for param in model.parameters()]

    criterion = nn.CrossEntropyLoss()
    if opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    elif opt == 'sgd':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif opt == 'momentum09':
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9)
    elif opt == 'nesterov09':
        optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=0.9, nesterov=True)
    elif opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=lr)
    t = tqdm.trange(max_batches)
    iter_loader = iter(train_loader)
    epoch_length = len(train_loader)
    epoch_losses_df = pd.DataFrame(columns=['Batch', 'Loss'])
    retrain_stats = pd.DataFrame(columns=['Batch', 'Validation Loss', 'Weight Distance'])
    scheduler = ExponentialLR(optimizer, gamma=0.95)
    tries = 0
    total_loss = 0.0
    start_loss = 1000000
    breaking = False
    model.to(device)
    model.train()
    for batch in range(max_batches):
        try:
            inputs, target_labels, _, _, _ = next(iter_loader)
        except StopIteration:
            iter_loader = iter(train_loader)
            inputs, target_labels, _, _, _ = next(iter_loader)
        inputs, target_labels = inputs.to(device), target_labels.to(device)

        outputs = model(inputs, None)
        loss = criterion(outputs, torch.argmax(target_labels, dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        t.update(1)

        if record_loss and ((batch % epoch_length == 0) or (batch == max_batches - 1)):
            total_loss += loss.item()

            scheduler.step()

            model.eval()
            with torch.inference_mode():
                test_loss = 0.0
                for inputs, target_labels, _, oracle_inputs, _ in test_loader:
                    inputs, target_labels, oracle_inputs = inputs.to(device), target_labels.to(device), oracle_inputs.to(device)
                    outputs = model(inputs, oracle_inputs)
                    loss = criterion(outputs, torch.argmax(target_labels, dim=1))
                    test_loss += loss.item()
                epoch_losses_df = epoch_losses_df.append({'Batch': batch, 'Loss': test_loss / len(test_loader)}, ignore_index=True)

            if batch % epoch_length == 0:
                avg_loss = test_loss / len(test_loader)

                if avg_loss >= start_loss - tolerance:
                    tries += 1
                    # start_loss = avg_loss - tolerance
                    print(f"Tries incremented: {tries}, Start Loss updated: {start_loss}, avg_loss was: {avg_loss}")
                    if tries >= max_tries:
                        breaking = True
                        print("break")
                else:
                    start_loss = avg_loss - tolerance
                    print('reset tries', start_loss)
                    tries = 0

                if save_models:
                    os.makedirs(save_path, exist_ok=True)
                    torch.save([model.kwargs, model.state_dict()], os.path.join(save_path, f'{repetition}-checkpoint-{batch // epoch_length}.pt'))

            model.train()

        if breaking:
            new_weights = [param.data.clone() for param in model.parameters()]
            weight_dist = weight_distance(old_weights, new_weights)
            retrain_stats = retrain_stats.append({'Batch': (batch // epoch_length) - tries, 'Validation Loss': avg_loss, 'Weight Distance': weight_dist}, ignore_index=True)
            break

    retrain_stats.to_csv(os.path.join(save_path, f'{opt}-stats-{repetition}.csv'), index=False)
    if record_loss:
        epoch_losses_df.to_csv(os.path.join(save_path, f'{opt}-losses-{repetition}.csv'), index=False)


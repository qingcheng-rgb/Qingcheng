def NN_training_module_shuffle(node_num,dt):
    run_number    = 1
    opexchange    = 'SPP'

    if int(run_number) == 1:
        data_location = 'training'
    else:
        data_location = 'secondRun'

    # node_num = 1005
    # dt = '2026-01-22'

    opexchange     = "SPP"
    data_location  = "training"  # 1=training, 2=secondRun
    gs_loc         = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"), columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist()
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr", value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))


    class DNN(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
            super(DNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size_1)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size_2, output_size)
        def forward(self, x):
            out = self.relu(self.fc1(x))
            out = self.relu2(self.fc2(out))
            return self.fc3(out)


    class QuantileDNN(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, quantiles):
            super(QuantileDNN, self).__init__()
            self.quantiles = quantiles
            self.output_size = output_size
            self.fc1 = nn.Linear(input_size, hidden_size_1)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size_2, output_size * len(quantiles))
        def forward(self, x):
            out = self.relu(self.fc1(x))
            out = self.relu2(self.fc2(out))
            x = self.fc3(out)
            return x.view(x.size(0), self.output_size, len(self.quantiles))


    class QuantileLoss(nn.Module):
        def __init__(self, quantiles):
            super(QuantileLoss, self).__init__()
            self.quantiles = quantiles
        def forward(self, predictions, targets):
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = targets - predictions[:, :, i]
                losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(2))
            return torch.mean(torch.cat(losses, dim=2))


    def predict_and_collect(model, data_loader, criterion, scaler_y=None):
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            if predictions.ndim == 3:
                num_samples, num_outputs, num_quantiles = predictions.shape
                transposed = np.transpose(predictions, (0, 2, 1)).reshape(-1, num_outputs)
                transposed = scaler_y.inverse_transform(transposed)
                predictions = np.transpose(transposed.reshape(num_samples, num_quantiles, num_outputs), (0, 2, 1))
            else:
                predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)


    quantiles      = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
    quantile_names = [f"q{int(q * 100)}" for q in quantiles]
    hidden_size_1  = 32
    hidden_size_2  = 8
    learning_rate  = 0.0001
    num_epochs     = 200
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_criterion     = nn.MSELoss()
    quantile_criterion = QuantileLoss(quantiles)

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min((pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")


        # 3-way split: 1/2 train, 1/4 validate, 1/4 test
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)
        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)
        X_train_tensor    = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_tensor    = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_validate_tensor = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_validate_tensor = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_tensor     = torch.tensor(xtest.values,    dtype=torch.float32).to(device)
        Y_test_tensor     = torch.tensor(ytest.values,    dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader    = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),       batch_size=128, shuffle=True, generator =g )
        validate_loader = DataLoader(TensorDataset(X_validate_tensor, Y_validate_tensor), batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor),         batch_size=128, shuffle=False)
        input_size, output_size = X_train_tensor.shape[1], Y_train_tensor.shape[1]
        mean_model     = DNN(input_size, hidden_size_1, hidden_size_2, output_size).to(device)
        quantile_model = QuantileDNN(input_size, hidden_size_1, hidden_size_2, output_size, quantiles).to(device)
        mean_optimizer     = optim.Adam(mean_model.parameters(),     lr=learning_rate)
        quantile_optimizer = optim.Adam(quantile_model.parameters(), lr=learning_rate)

        for model, optimizer, criterion in [(mean_model, mean_optimizer, mean_criterion),
                                            (quantile_model, quantile_optimizer, quantile_criterion)]:
            best_val_loss, patience, no_improve = float("inf"), 5, 0
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_loss = sum(criterion(model(inp), lbl).item() for inp, lbl in validate_loader) / len(validate_loader)
                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break


        mean_preds, _     = predict_and_collect(mean_model,     test_loader, mean_criterion,     scaler_y)
        quantile_preds, _ = predict_and_collect(quantile_model, test_loader, quantile_criterion, scaler_y)
        
        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df     = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])
        q_cols      = [f"{y}_{n}" for y in y_var for n in quantile_names]
        quantile_df = pd.DataFrame(quantile_preds.reshape(-1, output_size * len(quantiles)), columns=q_cols)

        result = pd.concat([ytest.reset_index(), mean_df, quantile_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = valuation_models.groupby(["dt", "hr", "node_num"]).max().reset_index()
    keep_cols = ["dt", "hr", "node_num", "da_total_mean", "rt_total_mean"] + \
                [f"da_total_{n}" for n in quantile_names] + [f"rt_total_{n}" for n in quantile_names]
    valuation_models["model"] = f"dnn_{hidden_size_1}h1_{hidden_size_2}h2"

    return valuation_models 

def NN_training_module_no_shuffle(node_num,dt):
    run_number    = 1
    opexchange    = 'SPP'

    if int(run_number) == 1:
        data_location = 'training'
    else:
        data_location = 'secondRun'

    # node_num = 1005
    # dt = '2026-01-22'

    opexchange     = "SPP"
    data_location  = "training"  # 1=training, 2=secondRun
    gs_loc         = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"), columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist()
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr", value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))


    class DNN(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size):
            super(DNN, self).__init__()
            self.fc1 = nn.Linear(input_size, hidden_size_1)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size_2, output_size)
        def forward(self, x):
            out = self.relu(self.fc1(x))
            out = self.relu2(self.fc2(out))
            return self.fc3(out)


    class QuantileDNN(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, quantiles):
            super(QuantileDNN, self).__init__()
            self.quantiles = quantiles
            self.output_size = output_size
            self.fc1 = nn.Linear(input_size, hidden_size_1)
            self.relu = nn.ReLU()
            self.fc2 = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.fc3 = nn.Linear(hidden_size_2, output_size * len(quantiles))
        def forward(self, x):
            out = self.relu(self.fc1(x))
            out = self.relu2(self.fc2(out))
            x = self.fc3(out)
            return x.view(x.size(0), self.output_size, len(self.quantiles))


    class QuantileLoss(nn.Module):
        def __init__(self, quantiles):
            super(QuantileLoss, self).__init__()
            self.quantiles = quantiles
        def forward(self, predictions, targets):
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = targets - predictions[:, :, i]
                losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(2))
            return torch.mean(torch.cat(losses, dim=2))


    def predict_and_collect(model, data_loader, criterion, scaler_y=None):
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            if predictions.ndim == 3:
                num_samples, num_outputs, num_quantiles = predictions.shape
                transposed = np.transpose(predictions, (0, 2, 1)).reshape(-1, num_outputs)
                transposed = scaler_y.inverse_transform(transposed)
                predictions = np.transpose(transposed.reshape(num_samples, num_quantiles, num_outputs), (0, 2, 1))
            else:
                predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)


    quantiles      = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
    quantile_names = [f"q{int(q * 100)}" for q in quantiles]
    hidden_size_1  = 32
    hidden_size_2  = 8
    learning_rate  = 0.0001
    num_epochs     = 200
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_criterion     = nn.MSELoss()
    quantile_criterion = QuantileLoss(quantiles)

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min((pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")

        # 3-way split: 1/2 train, 1/4 validate, 1/4 test
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)
        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)
        X_train_tensor    = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_tensor    = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_validate_tensor = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_validate_tensor = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_tensor     = torch.tensor(xtest.values,    dtype=torch.float32).to(device)
        Y_test_tensor     = torch.tensor(ytest.values,    dtype=torch.float32).to(device)
        train_loader    = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),       batch_size=128, shuffle=False)
        validate_loader = DataLoader(TensorDataset(X_validate_tensor, Y_validate_tensor), batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor),         batch_size=128, shuffle=False)
        input_size, output_size = X_train_tensor.shape[1], Y_train_tensor.shape[1]
        mean_model     = DNN(input_size, hidden_size_1, hidden_size_2, output_size).to(device)
        quantile_model = QuantileDNN(input_size, hidden_size_1, hidden_size_2, output_size, quantiles).to(device)
        mean_optimizer     = optim.Adam(mean_model.parameters(),     lr=learning_rate)
        quantile_optimizer = optim.Adam(quantile_model.parameters(), lr=learning_rate)

        for model, optimizer, criterion in [(mean_model, mean_optimizer, mean_criterion),
                                            (quantile_model, quantile_optimizer, quantile_criterion)]:
            best_val_loss, patience, no_improve = float("inf"), 5, 0
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_loss = sum(criterion(model(inp), lbl).item() for inp, lbl in validate_loader) / len(validate_loader)
                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break


        mean_preds, _     = predict_and_collect(mean_model,     test_loader, mean_criterion,     scaler_y)
        quantile_preds, _ = predict_and_collect(quantile_model, test_loader, quantile_criterion, scaler_y)
        
        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df     = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])
        q_cols      = [f"{y}_{n}" for y in y_var for n in quantile_names]
        quantile_df = pd.DataFrame(quantile_preds.reshape(-1, output_size * len(quantiles)), columns=q_cols)

        result = pd.concat([ytest.reset_index(), mean_df, quantile_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = valuation_models.groupby(["dt", "hr", "node_num"]).max().reset_index()
    keep_cols = ["dt", "hr", "node_num", "da_total_mean", "rt_total_mean"] + \
                [f"da_total_{n}" for n in quantile_names] + [f"rt_total_{n}" for n in quantile_names]
    valuation_models["model"] = f"dnn_{hidden_size_1}h1_{hidden_size_2}h2"

    return valuation_models 

def NN_training_with_pre_cnn(node_num, dt):

    run_number    = 1
    opexchange    = 'SPP'

    if int(run_number) == 1:
        data_location = 'training'
    else:
        data_location = 'secondRun'

    opexchange     = "SPP"
    data_location  = "training"
    gs_loc         = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"), columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist()
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr", value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))


    # ── model classes (all inside function) ─────────────────────

    class SpatialEncoder(nn.Module):
        def __init__(self, input_size, num_groups=10, conv_hidden=32, d_spatial=64):
            super(SpatialEncoder, self).__init__()
            self.num_groups = num_groups
            self.group_size = math.ceil(input_size / num_groups)
            self.padded_size = self.num_groups * self.group_size
            self.input_size = input_size
            self.conv1 = nn.Conv1d(self.group_size, conv_hidden, kernel_size=3, padding=1)
            self.bn1   = nn.BatchNorm1d(conv_hidden)
            self.conv2 = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=3, padding=1)
            self.bn2   = nn.BatchNorm1d(conv_hidden)
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.fc    = nn.Linear(conv_hidden, d_spatial)

        def forward(self, x):
            B = x.size(0)
            if self.input_size < self.padded_size:
                x = F.pad(x, (0, self.padded_size - self.input_size))
            x = x.view(B, self.num_groups, self.group_size).permute(0, 2, 1)
            x = self.bn1(F.relu(self.conv1(x)))
            x = self.bn2(F.relu(self.conv2(x)))
            x = self.pool(x).squeeze(-1)
            return self.fc(x)


    class SpatialDNN(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size,
                     num_groups=10, conv_hidden=32, d_spatial=64):
            super(SpatialDNN, self).__init__()
            self.spatial = SpatialEncoder(input_size, num_groups, conv_hidden, d_spatial)
            self.fc1  = nn.Linear(d_spatial, hidden_size_1)
            self.relu = nn.ReLU()
            self.fc2  = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.fc3  = nn.Linear(hidden_size_2, output_size)

        def forward(self, x):
            x = self.spatial(x)
            x = self.relu(self.fc1(x))
            x = self.relu2(self.fc2(x))
            return self.fc3(x)


    class SpatialQuantileDNN(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size,
                     quantiles, num_groups=10, conv_hidden=32, d_spatial=64):
            super(SpatialQuantileDNN, self).__init__()
            self.quantiles = quantiles
            self.output_size = output_size
            self.spatial = SpatialEncoder(input_size, num_groups, conv_hidden, d_spatial)
            self.fc1  = nn.Linear(d_spatial, hidden_size_1)
            self.relu = nn.ReLU()
            self.fc2  = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.fc3  = nn.Linear(hidden_size_2, output_size * len(quantiles))

        def forward(self, x):
            x = self.spatial(x)
            x = self.relu(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = self.fc3(x)
            return x.view(x.size(0), self.output_size, len(self.quantiles))


    class QuantileLoss(nn.Module):
        def __init__(self, quantiles):
            super(QuantileLoss, self).__init__()
            self.quantiles = quantiles
        def forward(self, predictions, targets):
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = targets - predictions[:, :, i]
                losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(2))
            return torch.mean(torch.cat(losses, dim=2))


    def predict_and_collect(model, data_loader, criterion, scaler_y=None):
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            if predictions.ndim == 3:
                num_samples, num_outputs, num_quantiles = predictions.shape
                transposed = np.transpose(predictions, (0, 2, 1)).reshape(-1, num_outputs)
                transposed = scaler_y.inverse_transform(transposed)
                predictions = np.transpose(transposed.reshape(num_samples, num_quantiles, num_outputs), (0, 2, 1))
            else:
                predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)


    # ── hyper-parameters ────────────────────────────────────────
    quantiles      = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
    quantile_names = [f"q{int(q * 100)}" for q in quantiles]
    hidden_size_1  = 32
    hidden_size_2  = 8
    learning_rate  = 0.0001
    num_epochs     = 200
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── spatial-encoder hyper-parameters ────────────────────────
    num_groups   = 20
    conv_hidden  = 64
    d_spatial    = 64

    mean_criterion     = nn.MSELoss()
    quantile_criterion = QuantileLoss(quantiles)

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min((pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")

        # 3-way split: 1/2 train, 1/4 validate, 1/4 test
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)
        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)
        X_train_tensor    = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_tensor    = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_validate_tensor = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_validate_tensor = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_tensor     = torch.tensor(xtest.values,     dtype=torch.float32).to(device)
        Y_test_tensor     = torch.tensor(ytest.values,     dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader    = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),       batch_size=128, shuffle=True, generator=g)
        validate_loader = DataLoader(TensorDataset(X_validate_tensor, Y_validate_tensor), batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor),         batch_size=128, shuffle=False)
        input_size, output_size = X_train_tensor.shape[1], Y_train_tensor.shape[1]

        # ── build models ────────────────────────────────────────
        mean_model = SpatialDNN(input_size, hidden_size_1, hidden_size_2, output_size,
                                num_groups=num_groups, conv_hidden=conv_hidden, d_spatial=d_spatial).to(device)
        quantile_model = SpatialQuantileDNN(input_size, hidden_size_1, hidden_size_2, output_size, quantiles,
                                            num_groups=num_groups, conv_hidden=conv_hidden, d_spatial=d_spatial).to(device)
        mean_optimizer     = optim.Adam(mean_model.parameters(),     lr=learning_rate)
        quantile_optimizer = optim.Adam(quantile_model.parameters(), lr=learning_rate)

        for model, optimizer, criterion in [(mean_model, mean_optimizer, mean_criterion),
                                            (quantile_model, quantile_optimizer, quantile_criterion)]:
            best_val_loss, patience, no_improve = float("inf"), 5, 0
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_loss = sum(criterion(model(inp), lbl).item() for inp, lbl in validate_loader) / len(validate_loader)
                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        mean_preds, _     = predict_and_collect(mean_model,     test_loader, mean_criterion,     scaler_y)
        quantile_preds, _ = predict_and_collect(quantile_model, test_loader, quantile_criterion, scaler_y)

        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df     = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])
        q_cols      = [f"{y}_{n}" for y in y_var for n in quantile_names]
        quantile_df = pd.DataFrame(quantile_preds.reshape(-1, output_size * len(quantiles)), columns=q_cols)

        result = pd.concat([ytest.reset_index(), mean_df, quantile_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = valuation_models.groupby(["dt", "hr", "node_num"]).max().reset_index()
    keep_cols = ["dt", "hr", "node_num", "da_total_mean", "rt_total_mean"] + \
                [f"da_total_{n}" for n in quantile_names] + [f"rt_total_{n}" for n in quantile_names]
    valuation_models["model"] = f"spatial_cnn_{conv_hidden}conv_{d_spatial}emb_{hidden_size_1}h1_{hidden_size_2}h2"

    return valuation_models

def NN_training_with_post_cnn(node_num, dt):
    import torch.nn.functional as F

    # Architecture: Input -> FC1 -> FC2 (hidden rep) -> CNN over hidden units -> output
    # CNN treats the hidden_size_2-dim vector as a 1-D sequence,
    # capturing local structure among learned hidden features.

    class PostCNNDNN(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, conv_hidden=32):
            super().__init__()
            self.fc1   = nn.Linear(input_size, hidden_size_1)
            self.relu  = nn.ReLU()
            self.fc2   = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.conv  = nn.Conv1d(1, conv_hidden, kernel_size=3, padding=1)
            self.bn    = nn.BatchNorm1d(conv_hidden)
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.fc3   = nn.Linear(conv_hidden, output_size)

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu2(self.fc2(x))        # (B, hidden_size_2)
            x = x.unsqueeze(1)                 # (B, 1, hidden_size_2)
            x = self.bn(F.relu(self.conv(x)))  # (B, conv_hidden, hidden_size_2)
            x = self.pool(x).squeeze(-1)       # (B, conv_hidden)
            return self.fc3(x)

    class PostCNNQuantileDNN(nn.Module):
        def __init__(self, input_size, hidden_size_1, hidden_size_2, output_size, quantiles, conv_hidden=32):
            super().__init__()
            self.quantiles   = quantiles
            self.output_size = output_size
            self.fc1   = nn.Linear(input_size, hidden_size_1)
            self.relu  = nn.ReLU()
            self.fc2   = nn.Linear(hidden_size_1, hidden_size_2)
            self.relu2 = nn.ReLU()
            self.conv  = nn.Conv1d(1, conv_hidden, kernel_size=3, padding=1)
            self.bn    = nn.BatchNorm1d(conv_hidden)
            self.pool  = nn.AdaptiveAvgPool1d(1)
            self.fc3   = nn.Linear(conv_hidden, output_size * len(quantiles))

        def forward(self, x):
            x = self.relu(self.fc1(x))
            x = self.relu2(self.fc2(x))
            x = x.unsqueeze(1)
            x = self.bn(F.relu(self.conv(x)))
            x = self.pool(x).squeeze(-1)
            x = self.fc3(x)
            return x.view(x.size(0), self.output_size, len(self.quantiles))

    class QuantileLoss(nn.Module):
        def __init__(self, quantiles):
            super().__init__()
            self.quantiles = quantiles
        def forward(self, predictions, targets):
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = targets - predictions[:, :, i]
                losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(2))
            return torch.mean(torch.cat(losses, dim=2))

    def predict_and_collect(model, data_loader, criterion, scaler_y=None, normalize=True):
        model.eval()
        total_loss  = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                dev = next(model.parameters()).device
                inputs, labels = inputs.to(dev), labels.to(dev)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            if predictions.ndim == 3:
                ns, no, nq = predictions.shape
                t = np.transpose(predictions, (0, 2, 1)).reshape(-1, no)
                t = scaler_y.inverse_transform(t)
                predictions = np.transpose(t.reshape(ns, nq, no), (0, 2, 1))
            else:
                predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)

    opexchange    = "SPP"
    data_location = "training"
    gs_loc        = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns
                        if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = (pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"),
                     columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist())
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr",
                       value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))

    quantiles      = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4,
                      0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
    quantile_names = [f"q{int(q * 100)}" for q in quantiles]
    hidden_size_1  = 32
    hidden_size_2  = 8
    conv_hidden    = 32
    learning_rate  = 0.0001
    num_epochs     = 200
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_criterion     = nn.MSELoss()
    quantile_criterion = QuantileLoss(quantiles)

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min(
            (pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")

        xtrain, xtest,     ytrain, ytest     = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)

        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)

        X_train_t = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_t = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_val_t   = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_val_t   = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_t  = torch.tensor(xtest.values,     dtype=torch.float32).to(device)
        Y_test_t  = torch.tensor(ytest.values,     dtype=torch.float32).to(device)

        g = torch.Generator(); g.manual_seed(42)
        train_loader    = DataLoader(TensorDataset(X_train_t, Y_train_t), batch_size=128, shuffle=True,  generator=g)
        validate_loader = DataLoader(TensorDataset(X_val_t,   Y_val_t),   batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_t,  Y_test_t),  batch_size=128, shuffle=False)

        input_size  = X_train_t.shape[1]
        output_size = Y_train_t.shape[1]

        mean_model     = PostCNNDNN(input_size, hidden_size_1, hidden_size_2, output_size, conv_hidden).to(device)
        quantile_model = PostCNNQuantileDNN(input_size, hidden_size_1, hidden_size_2, output_size, quantiles, conv_hidden).to(device)
        mean_optimizer     = optim.Adam(mean_model.parameters(),     lr=learning_rate)
        quantile_optimizer = optim.Adam(quantile_model.parameters(), lr=learning_rate)

        for model, optimizer, criterion in [
            (mean_model,     mean_optimizer,     mean_criterion),
            (quantile_model, quantile_optimizer, quantile_criterion),
        ]:
            best_val_loss, patience, no_improve = float("inf"), 5, 0
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_loss = sum(criterion(model(inp.to(device)), lbl.to(device)).item()
                               for inp, lbl in validate_loader) / len(validate_loader)
                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        mean_preds,     _ = predict_and_collect(mean_model,     test_loader, mean_criterion,     scaler_y, normalize)
        quantile_preds, _ = predict_and_collect(quantile_model, test_loader, quantile_criterion, scaler_y, normalize)

        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df     = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])
        q_cols      = [f"{y}_{n}" for y in y_var for n in quantile_names]
        quantile_df = pd.DataFrame(quantile_preds.reshape(-1, output_size * len(quantiles)), columns=q_cols)

        result = pd.concat([ytest.reset_index(), mean_df, quantile_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = (valuation_models
                        .groupby(["dt", "hr", "node_num"]).max().reset_index())
    valuation_models["model"] = f"post_cnn_{conv_hidden}conv_{hidden_size_1}h1_{hidden_size_2}h2"
    return valuation_models

def pure_CNN(node_num, dt):

    run_number    = 1
    opexchange    = 'SPP'

    if int(run_number) == 1:
        data_location = 'training'
    else:
        data_location = 'secondRun'

    opexchange     = "SPP"
    data_location  = "training"
    gs_loc         = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"), columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist()
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr", value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))


    # ── model classes (all inside function) ─────────────────────

    class PureCNN(nn.Module):
        """
        Fully-convolutional network for mean prediction.
        No nn.Linear layers — the final Conv1d with kernel_size=1 acts as the
        prediction head, followed by global average pooling.
        """
        def __init__(self, input_size, output_size, conv_hidden=64, kernel_size=3):
            super(PureCNN, self).__init__()
            self.input_size = input_size
            self.conv1 = nn.Conv1d(1, conv_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn1   = nn.BatchNorm1d(conv_hidden)
            self.conv2 = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn2   = nn.BatchNorm1d(conv_hidden)
            self.conv3 = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn3   = nn.BatchNorm1d(conv_hidden)
            # Prediction head: 1x1 conv maps conv_hidden channels → output_size channels
            self.pred_conv = nn.Conv1d(conv_hidden, output_size, kernel_size=1)
            self.pool      = nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            # x shape: (B, input_size)
            x = x.unsqueeze(1)                    # (B, 1, input_size)
            x = self.bn1(F.relu(self.conv1(x)))   # (B, conv_hidden, input_size)
            x = self.bn2(F.relu(self.conv2(x)))   # (B, conv_hidden, input_size)
            x = self.bn3(F.relu(self.conv3(x)))   # (B, conv_hidden, input_size)
            x = self.pred_conv(x)                 # (B, output_size, input_size)
            x = self.pool(x).squeeze(-1)          # (B, output_size)
            return x


    class PureQuantileCNN(nn.Module):
        """
        Fully-convolutional network for quantile prediction.
        Final 1x1 conv outputs output_size * num_quantiles channels, reshaped at the end.
        """
        def __init__(self, input_size, output_size, quantiles, conv_hidden=64, kernel_size=3):
            super(PureQuantileCNN, self).__init__()
            self.input_size = input_size
            self.output_size = output_size
            self.quantiles = quantiles
            self.conv1 = nn.Conv1d(1, conv_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn1   = nn.BatchNorm1d(conv_hidden)
            self.conv2 = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn2   = nn.BatchNorm1d(conv_hidden)
            self.conv3 = nn.Conv1d(conv_hidden, conv_hidden, kernel_size=kernel_size, padding=kernel_size // 2)
            self.bn3   = nn.BatchNorm1d(conv_hidden)
            # Prediction head: 1x1 conv → output_size * num_quantiles channels
            self.pred_conv = nn.Conv1d(conv_hidden, output_size * len(quantiles), kernel_size=1)
            self.pool      = nn.AdaptiveAvgPool1d(1)

        def forward(self, x):
            # x shape: (B, input_size)
            x = x.unsqueeze(1)                    # (B, 1, input_size)
            x = self.bn1(F.relu(self.conv1(x)))   # (B, conv_hidden, input_size)
            x = self.bn2(F.relu(self.conv2(x)))   # (B, conv_hidden, input_size)
            x = self.bn3(F.relu(self.conv3(x)))   # (B, conv_hidden, input_size)
            x = self.pred_conv(x)                 # (B, output_size*num_q, input_size)
            x = self.pool(x).squeeze(-1)          # (B, output_size*num_q)
            return x.view(x.size(0), self.output_size, len(self.quantiles))


    class QuantileLoss(nn.Module):
        def __init__(self, quantiles):
            super(QuantileLoss, self).__init__()
            self.quantiles = quantiles
        def forward(self, predictions, targets):
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = targets - predictions[:, :, i]
                losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(2))
            return torch.mean(torch.cat(losses, dim=2))


    def predict_and_collect(model, data_loader, criterion, scaler_y=None):
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            if predictions.ndim == 3:
                num_samples, num_outputs, num_quantiles = predictions.shape
                transposed = np.transpose(predictions, (0, 2, 1)).reshape(-1, num_outputs)
                transposed = scaler_y.inverse_transform(transposed)
                predictions = np.transpose(transposed.reshape(num_samples, num_quantiles, num_outputs), (0, 2, 1))
            else:
                predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)


    # ── hyper-parameters ────────────────────────────────────────
    quantiles      = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
    quantile_names = [f"q{int(q * 100)}" for q in quantiles]
    learning_rate  = 0.0001
    num_epochs     = 200
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Pure CNN hyper-parameters ───────────────────────────────
    conv_hidden  = 64
    kernel_size  = 3

    mean_criterion     = nn.MSELoss()
    quantile_criterion = QuantileLoss(quantiles)

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min((pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")

        # 3-way split: 1/2 train, 1/4 validate, 1/4 test
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)
        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)
        X_train_tensor    = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_tensor    = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_validate_tensor = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_validate_tensor = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_tensor     = torch.tensor(xtest.values,     dtype=torch.float32).to(device)
        Y_test_tensor     = torch.tensor(ytest.values,     dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader    = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),       batch_size=128, shuffle=True, generator=g)
        validate_loader = DataLoader(TensorDataset(X_validate_tensor, Y_validate_tensor), batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor),         batch_size=128, shuffle=False)
        input_size, output_size = X_train_tensor.shape[1], Y_train_tensor.shape[1]

        # ── build models ────────────────────────────────────────
        mean_model     = PureCNN(input_size, output_size,
                                 conv_hidden=conv_hidden, kernel_size=kernel_size).to(device)
        quantile_model = PureQuantileCNN(input_size, output_size, quantiles,
                                         conv_hidden=conv_hidden, kernel_size=kernel_size).to(device)
        mean_optimizer     = optim.Adam(mean_model.parameters(),     lr=learning_rate)
        quantile_optimizer = optim.Adam(quantile_model.parameters(), lr=learning_rate)

        for model, optimizer, criterion in [(mean_model, mean_optimizer, mean_criterion),
                                            (quantile_model, quantile_optimizer, quantile_criterion)]:
            best_val_loss, patience, no_improve = float("inf"), 5, 0
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_loss = sum(criterion(model(inp), lbl).item() for inp, lbl in validate_loader) / len(validate_loader)
                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        mean_preds, _     = predict_and_collect(mean_model,     test_loader, mean_criterion,     scaler_y)
        quantile_preds, _ = predict_and_collect(quantile_model, test_loader, quantile_criterion, scaler_y)

        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df     = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])
        q_cols      = [f"{y}_{n}" for y in y_var for n in quantile_names]
        quantile_df = pd.DataFrame(quantile_preds.reshape(-1, output_size * len(quantiles)), columns=q_cols)

        result = pd.concat([ytest.reset_index(), mean_df, quantile_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = valuation_models.groupby(["dt", "hr", "node_num"]).max().reset_index()
    keep_cols = ["dt", "hr", "node_num", "da_total_mean", "rt_total_mean"] + \
                [f"da_total_{n}" for n in quantile_names] + [f"rt_total_{n}" for n in quantile_names]
    valuation_models["model"] = f"pure_cnn_{conv_hidden}conv_k{kernel_size}"

    return valuation_models

def GRU_framework(node_num, dt):

    run_number    = 1
    opexchange    = 'SPP'

    if int(run_number) == 1:
        data_location = 'training'
    else:
        data_location = 'secondRun'

    opexchange     = "SPP"
    data_location  = "training"
    gs_loc         = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"), columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist()
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr", value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))


    # ── model classes (all inside function) ─────────────────────

    class GRUMean(nn.Module):
        """
        GRU-based network for mean prediction.
        Treats the flat feature vector as a length-input_size sequence with 1 feature per step,
        runs a bidirectional multi-layer GRU, then projects the final hidden state to outputs.
        """
        def __init__(self, input_size, output_size, hidden_size=64, num_layers=2,
                     bidirectional=True, dropout=0.1):
            super(GRUMean, self).__init__()
            self.hidden_size   = hidden_size
            self.num_layers    = num_layers
            self.bidirectional = bidirectional
            self.gru = nn.GRU(
                input_size=1,                  # 1 value per sequence step
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # If bidirectional, the GRU outputs 2 * hidden_size features per step
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.fc = nn.Linear(out_dim, output_size)

        def forward(self, x):
            # x shape: (B, input_size)
            x = x.unsqueeze(-1)                   # (B, input_size, 1) — seq_len=input_size, features=1
            out, h_n = self.gru(x)                # out: (B, input_size, hidden_size * num_directions)
            # Use the last time step's output (contains both directions if bidirectional)
            last = out[:, -1, :]                  # (B, hidden_size * num_directions)
            return self.fc(last)                  # (B, output_size)


    class GRUQuantile(nn.Module):
        """GRU-based network for quantile prediction."""
        def __init__(self, input_size, output_size, quantiles, hidden_size=64, num_layers=2,
                     bidirectional=True, dropout=0.1):
            super(GRUQuantile, self).__init__()
            self.output_size   = output_size
            self.quantiles     = quantiles
            self.hidden_size   = hidden_size
            self.num_layers    = num_layers
            self.bidirectional = bidirectional
            self.gru = nn.GRU(
                input_size=1,
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.fc = nn.Linear(out_dim, output_size * len(quantiles))

        def forward(self, x):
            # x shape: (B, input_size)
            x = x.unsqueeze(-1)                   # (B, input_size, 1)
            out, h_n = self.gru(x)                # (B, input_size, hidden_size * num_directions)
            last = out[:, -1, :]                  # (B, hidden_size * num_directions)
            x = self.fc(last)                     # (B, output_size * num_quantiles)
            return x.view(x.size(0), self.output_size, len(self.quantiles))


    class QuantileLoss(nn.Module):
        def __init__(self, quantiles):
            super(QuantileLoss, self).__init__()
            self.quantiles = quantiles
        def forward(self, predictions, targets):
            losses = []
            for i, q in enumerate(self.quantiles):
                errors = targets - predictions[:, :, i]
                losses.append(torch.max((q - 1) * errors, q * errors).unsqueeze(2))
            return torch.mean(torch.cat(losses, dim=2))


    def predict_and_collect(model, data_loader, criterion, scaler_y=None):
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            if predictions.ndim == 3:
                num_samples, num_outputs, num_quantiles = predictions.shape
                transposed = np.transpose(predictions, (0, 2, 1)).reshape(-1, num_outputs)
                transposed = scaler_y.inverse_transform(transposed)
                predictions = np.transpose(transposed.reshape(num_samples, num_quantiles, num_outputs), (0, 2, 1))
            else:
                predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)


    # ── hyper-parameters ────────────────────────────────────────
    quantiles      = [0.01, 0.03, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97, 0.99]
    quantile_names = [f"q{int(q * 100)}" for q in quantiles]
    learning_rate  = 0.0001
    num_epochs     = 200
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── GRU hyper-parameters ────────────────────────────────────
    hidden_size   = 64
    num_layers    = 2
    bidirectional = True
    dropout       = 0.1

    mean_criterion     = nn.MSELoss()
    quantile_criterion = QuantileLoss(quantiles)

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min((pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")

        # 3-way split: 1/2 train, 1/4 validate, 1/4 test
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)
        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)
        X_train_tensor    = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_tensor    = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_validate_tensor = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_validate_tensor = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_tensor     = torch.tensor(xtest.values,     dtype=torch.float32).to(device)
        Y_test_tensor     = torch.tensor(ytest.values,     dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader    = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),       batch_size=128, shuffle=True, generator=g)
        validate_loader = DataLoader(TensorDataset(X_validate_tensor, Y_validate_tensor), batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor),         batch_size=128, shuffle=False)
        input_size, output_size = X_train_tensor.shape[1], Y_train_tensor.shape[1]

        # ── build models ────────────────────────────────────────
        mean_model = GRUMean(input_size, output_size,
                             hidden_size=hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=dropout).to(device)
        quantile_model = GRUQuantile(input_size, output_size, quantiles,
                                     hidden_size=hidden_size, num_layers=num_layers,
                                     bidirectional=bidirectional, dropout=dropout).to(device)
        mean_optimizer     = optim.Adam(mean_model.parameters(),     lr=learning_rate)
        quantile_optimizer = optim.Adam(quantile_model.parameters(), lr=learning_rate)

        for model, optimizer, criterion in [(mean_model, mean_optimizer, mean_criterion),
                                            (quantile_model, quantile_optimizer, quantile_criterion)]:
            best_val_loss, patience, no_improve = float("inf"), 5, 0
            for epoch in range(num_epochs):
                model.train()
                for inputs, labels in train_loader:
                    optimizer.zero_grad()
                    loss = criterion(model(inputs), labels)
                    loss.backward()
                    optimizer.step()
                model.eval()
                val_loss = sum(criterion(model(inp), lbl).item() for inp, lbl in validate_loader) / len(validate_loader)
                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                else:
                    no_improve += 1
                    if no_improve >= patience:
                        print(f"Early stopping at epoch {epoch + 1}")
                        break

        mean_preds, _     = predict_and_collect(mean_model,     test_loader, mean_criterion,     scaler_y)
        quantile_preds, _ = predict_and_collect(quantile_model, test_loader, quantile_criterion, scaler_y)

        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df     = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])
        q_cols      = [f"{y}_{n}" for y in y_var for n in quantile_names]
        quantile_df = pd.DataFrame(quantile_preds.reshape(-1, output_size * len(quantiles)), columns=q_cols)

        result = pd.concat([ytest.reset_index(), mean_df, quantile_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = valuation_models.groupby(["dt", "hr", "node_num"]).max().reset_index()
    keep_cols = ["dt", "hr", "node_num", "da_total_mean", "rt_total_mean"] + \
                [f"da_total_{n}" for n in quantile_names] + [f"rt_total_{n}" for n in quantile_names]
    direction_tag = "bi" if bidirectional else "uni"
    valuation_models["model"] = f"gru_{direction_tag}_{hidden_size}h_{num_layers}L"

    return valuation_models

def GRU_framework_only_means(node_num, dt):

    run_number    = 1
    opexchange    = 'SPP'

    if int(run_number) == 1:
        data_location = 'training'
    else:
        data_location = 'secondRun'

    opexchange     = "SPP"
    data_location  = "training"
    gs_loc         = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"), columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist()
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr", value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))


    # ── model classes (all inside function) ─────────────────────

    class GRUMean(nn.Module):
        """
        GRU-based network for mean prediction.
        Treats the flat feature vector as a length-input_size sequence with 1 feature per step,
        runs a bidirectional multi-layer GRU, then projects the final hidden state to outputs.
        """
        def __init__(self, input_size, output_size, hidden_size=64, num_layers=2,
                     bidirectional=True, dropout=0.1):
            super(GRUMean, self).__init__()
            self.hidden_size   = hidden_size
            self.num_layers    = num_layers
            self.bidirectional = bidirectional
            self.gru = nn.GRU(
                input_size=1,                  # 1 value per sequence step
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # If bidirectional, the GRU outputs 2 * hidden_size features per step
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.fc = nn.Linear(out_dim, output_size)

        def forward(self, x):
            # x shape: (B, input_size)
            x = x.unsqueeze(-1)                   # (B, input_size, 1) — seq_len=input_size, features=1
            out, h_n = self.gru(x)                # out: (B, input_size, hidden_size * num_directions)
            # Use the last time step's output (contains both directions if bidirectional)
            last = out[:, -1, :]                  # (B, hidden_size * num_directions)
            return self.fc(last)                  # (B, output_size)


    def predict_and_collect(model, data_loader, criterion, scaler_y=None):
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)


    # ── hyper-parameters ────────────────────────────────────────
    learning_rate  = 0.01
    num_epochs     = 30
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── GRU hyper-parameters ────────────────────────────────────
    hidden_size   = 32
    num_layers    = 1
    bidirectional = True
    dropout       = 0.1

    mean_criterion = nn.MSELoss()

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min((pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")

        # 3-way split: 1/2 train, 1/4 validate, 1/4 test
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)
        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)
        X_train_tensor    = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_tensor    = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_validate_tensor = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_validate_tensor = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_tensor     = torch.tensor(xtest.values,     dtype=torch.float32).to(device)
        Y_test_tensor     = torch.tensor(ytest.values,     dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader    = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),       batch_size=128, shuffle=True, generator=g)
        validate_loader = DataLoader(TensorDataset(X_validate_tensor, Y_validate_tensor), batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor),         batch_size=128, shuffle=False)
        input_size, output_size = X_train_tensor.shape[1], Y_train_tensor.shape[1]

        # ── build model ─────────────────────────────────────────
        mean_model = GRUMean(input_size, output_size,
                             hidden_size=hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=dropout).to(device)
        mean_optimizer = optim.Adam(mean_model.parameters(), lr=learning_rate)

        # ── train mean model ────────────────────────────────────
        best_val_loss, patience, no_improve = float("inf"), 5, 0
        for epoch in range(num_epochs):
            mean_model.train()
            for inputs, labels in train_loader:
                mean_optimizer.zero_grad()
                loss = mean_criterion(mean_model(inputs), labels)
                loss.backward()
                mean_optimizer.step()
            mean_model.eval()
            val_loss = sum(mean_criterion(mean_model(inp), lbl).item() for inp, lbl in validate_loader) / len(validate_loader)
            if val_loss < best_val_loss:
                best_val_loss, no_improve = val_loss, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        mean_preds, _ = predict_and_collect(mean_model, test_loader, mean_criterion, scaler_y)

        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])

        result = pd.concat([ytest.reset_index(), mean_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = valuation_models.groupby(["dt", "hr", "node_num"]).max().reset_index()
    keep_cols = ["dt", "hr", "node_num", "da_total_mean", "rt_total_mean"]
    direction_tag = "bi" if bidirectional else "uni"
    valuation_models["model"] = f"gru_{direction_tag}_{hidden_size}h_{num_layers}L_mean_only"

    return valuation_models

def GRU_framework_means_print(node_num, dt):

    run_number    = 1
    opexchange    = 'SPP'

    if int(run_number) == 1:
        data_location = 'training'
    else:
        data_location = 'secondRun'

    opexchange     = "SPP"
    data_location  = "training"
    gs_loc         = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"), columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist()
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr", value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))


    # ── model classes (all inside function) ─────────────────────

    class GRUMean(nn.Module):
        """
        GRU-based network for mean prediction.
        Treats the flat feature vector as a length-input_size sequence with 1 feature per step,
        runs a bidirectional multi-layer GRU, then projects the final hidden state to outputs.
        """
        def __init__(self, input_size, output_size, hidden_size=16, num_layers=2,
                     bidirectional=True, dropout=0.1):
            super(GRUMean, self).__init__()
            self.hidden_size   = hidden_size
            self.num_layers    = num_layers
            self.bidirectional = bidirectional
            self.gru = nn.GRU(
                input_size=1,                  # 1 value per sequence step
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.fc = nn.Linear(out_dim, output_size)

        def forward(self, x):
            x = x.unsqueeze(-1)
            out, h_n = self.gru(x)
            last = out.mean(dim=1)
            return self.fc(last)


    def predict_and_collect(model, data_loader, criterion, scaler_y=None):
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)


    # ── diagnostic helpers ──────────────────────────────────────

    def diagnose_hidden_trajectories(model, x_sample, top_k=5):
        """
        Run one example through the GRU and report which hidden dimensions stay flat
        (long-term memory) vs. oscillate (short-term features).

        Measures "flatness" via the standard deviation of each dimension's trajectory
        across the sequence. Low std = flat = long-term memory.
        """
        print("\n" + "=" * 72)
        print("DIAGNOSTIC 1: Hidden state trajectories")
        print("=" * 72)
        model.eval()
        with torch.no_grad():
            x = x_sample.unsqueeze(0).unsqueeze(-1).to(device)   # (1, seq_len, 1)
            out, h_n = model.gru(x)                              # (1, seq_len, hidden * dirs)
            out_np = out[0].cpu().numpy()                        # (seq_len, hidden * dirs)

        seq_len, total_hidden = out_np.shape
        stds  = out_np.std(axis=0)                               # per-dimension std across positions
        means = out_np.mean(axis=0)

        print(f"Sequence length (features): {seq_len}")
        print(f"Total hidden dims (incl. directions): {total_hidden}")
        print(f"Overall std of activations: mean={stds.mean():.4f}, "
              f"min={stds.min():.4f}, max={stds.max():.4f}")

        flat_idx   = np.argsort(stds)[:top_k]                    # lowest std = flattest
        active_idx = np.argsort(stds)[-top_k:][::-1]             # highest std = most active

        print(f"\nTop {top_k} FLATTEST dimensions (long-term memory candidates):")
        for k in flat_idx:
            traj = out_np[:, k]
            print(f"  dim {k:3d}:  std={stds[k]:.4f}  mean={means[k]:+.4f}  "
                  f"start={traj[0]:+.3f}  mid={traj[seq_len//2]:+.3f}  end={traj[-1]:+.3f}")

        print(f"\nTop {top_k} MOST ACTIVE dimensions (short-term / oscillating):")
        for k in active_idx:
            traj = out_np[:, k]
            print(f"  dim {k:3d}:  std={stds[k]:.4f}  mean={means[k]:+.4f}  "
                  f"min={traj.min():+.3f}  max={traj.max():+.3f}  range={traj.max()-traj.min():.3f}")

        # Coarse ASCII plot for the single most active dim
        print(f"\nASCII trajectory of most active dim ({active_idx[0]}):")
        k = active_idx[0]
        traj = out_np[:, k]
        lo, hi = traj.min(), traj.max()
        norm = (traj - lo) / (hi - lo + 1e-9)
        rows = 8
        for r in range(rows, 0, -1):
            threshold = r / rows
            line = "".join("█" if v >= threshold - 1/rows else " " for v in norm)
            print(f"  {line}")
        print(f"  pos 0{' ' * (seq_len - 8)}pos {seq_len - 1}")


    def diagnose_gate_values(model, x_sample):
        """
        Extract forward-direction update (z) and reset (r) gates at every position
        by replaying layer 1 of the GRU with a GRUCell using the same weights.
        """
        print("\n" + "=" * 72)
        print("DIAGNOSTIC 2: Gate values (update gate z, reset gate r)")
        print("=" * 72)
        model.eval()

        # Pull the forward-direction layer-0 weights out of the GRU module
        # PyTorch stores them as:  weight_ih_l0, weight_hh_l0, bias_ih_l0, bias_hh_l0
        W_ih = model.gru.weight_ih_l0.data       # (3*hidden, input_size)
        W_hh = model.gru.weight_hh_l0.data       # (3*hidden, hidden)
        b_ih = model.gru.bias_ih_l0.data         # (3*hidden,)
        b_hh = model.gru.bias_hh_l0.data         # (3*hidden,)
        hidden_size = model.hidden_size

        # Slice the 3 gates out of the stacked weight matrix
        # PyTorch GRU convention: gates are ordered [r (reset), z (update), n (new/candidate)]
        W_ir, W_iz, W_in = W_ih[:hidden_size], W_ih[hidden_size:2*hidden_size], W_ih[2*hidden_size:]
        W_hr, W_hz, W_hn = W_hh[:hidden_size], W_hh[hidden_size:2*hidden_size], W_hh[2*hidden_size:]
        b_ir, b_iz, b_in = b_ih[:hidden_size], b_ih[hidden_size:2*hidden_size], b_ih[2*hidden_size:]
        b_hr, b_hz, b_hn = b_hh[:hidden_size], b_hh[hidden_size:2*hidden_size], b_hh[2*hidden_size:]

        with torch.no_grad():
            x = x_sample.unsqueeze(0).unsqueeze(-1).to(device)   # (1, seq_len, 1)
            seq_len = x.size(1)
            h = torch.zeros(1, hidden_size, device=device)
            z_hist = torch.zeros(seq_len, hidden_size, device=device)
            r_hist = torch.zeros(seq_len, hidden_size, device=device)

            for t in range(seq_len):
                x_t = x[:, t, :]                                  # (1, 1)
                r_t = torch.sigmoid(x_t @ W_ir.T + b_ir + h @ W_hr.T + b_hr)
                z_t = torch.sigmoid(x_t @ W_iz.T + b_iz + h @ W_hz.T + b_hz)
                n_t = torch.tanh(x_t @ W_in.T + b_in + r_t * (h @ W_hn.T + b_hn))
                h   = (1 - z_t) * n_t + z_t * h
                z_hist[t] = z_t[0]
                r_hist[t] = r_t[0]

        z_np = z_hist.cpu().numpy()
        r_np = r_hist.cpu().numpy()

        print(f"Update gate z — overall mean={z_np.mean():.3f}, std={z_np.std():.3f}")
        print(f"Reset  gate r — overall mean={r_np.mean():.3f}, std={r_np.std():.3f}")
        print("(z ≈ 1 → KEEP old memory, z ≈ 0 → OVERWRITE with new candidate)")
        print("(r ≈ 1 → USE old memory for candidate, r ≈ 0 → IGNORE old memory)")

        # Per-dimension summary: average z across the sequence
        z_per_dim = z_np.mean(axis=0)
        r_per_dim = r_np.mean(axis=0)

        print("\nPer-dimension AVG update gate z (sorted):")
        order = np.argsort(z_per_dim)
        print("  Lowest-z dims (actively overwriting → short-term):")
        for k in order[:5]:
            print(f"    dim {k:3d}:  avg z={z_per_dim[k]:.3f}  avg r={r_per_dim[k]:.3f}")
        print("  Highest-z dims (locking in memory → long-term):")
        for k in order[-5:][::-1]:
            print(f"    dim {k:3d}:  avg z={z_per_dim[k]:.3f}  avg r={r_per_dim[k]:.3f}")

        # How many dims are "mostly locked" vs "mostly overwriting"?
        locked      = (z_per_dim > 0.7).sum()
        overwriting = (z_per_dim < 0.3).sum()
        middle      = hidden_size - locked - overwriting
        print(f"\nGating regime breakdown across {hidden_size} dims:")
        print(f"  mostly KEEPING memory (avg z > 0.7):       {locked:3d} dims")
        print(f"  mostly OVERWRITING memory (avg z < 0.3):   {overwriting:3d} dims")
        print(f"  mixed / dynamic (0.3 ≤ avg z ≤ 0.7):       {middle:3d} dims")


    def diagnose_position_ablation(model, x_sample, scaler_y=None, group_size=10):
        """
        Zero out each position (in groups) of the input and measure prediction change.
        Larger change = that position contributes more to the prediction.
        """
        print("\n" + "=" * 72)
        print("DIAGNOSTIC 3: Position ablation (which positions matter most?)")
        print("=" * 72)
        model.eval()
        with torch.no_grad():
            x = x_sample.unsqueeze(0).to(device)                  # (1, seq_len)
            baseline = model(x).cpu().numpy()[0]                  # (output_size,)
            seq_len = x.size(1)

            deltas = []
            num_groups = math.ceil(seq_len / group_size)
            for g in range(num_groups):
                start = g * group_size
                end   = min(start + group_size, seq_len)
                x_ab = x.clone()
                x_ab[:, start:end] = 0.0
                pred = model(x_ab).cpu().numpy()[0]
                change = np.abs(pred - baseline).mean()
                deltas.append((start, end, change))

        changes = np.array([d[2] for d in deltas])
        print(f"Ablation groups of {group_size} positions each, {num_groups} groups total.")
        print(f"Baseline prediction (scaled): {baseline}")
        print(f"Change from zeroing: mean={changes.mean():.4f}, "
              f"min={changes.min():.4f}, max={changes.max():.4f}")

        # Top 5 most impactful position groups
        top_idx = np.argsort(changes)[-5:][::-1]
        bot_idx = np.argsort(changes)[:5]

        print(f"\nTop 5 MOST IMPACTFUL position groups (big prediction change when zeroed):")
        for i in top_idx:
            s, e, c = deltas[i]
            print(f"  positions {s:3d}–{e-1:3d}:  change = {c:.4f}")

        print(f"\nTop 5 LEAST IMPACTFUL position groups (prediction barely changes):")
        for i in bot_idx:
            s, e, c = deltas[i]
            print(f"  positions {s:3d}–{e-1:3d}:  change = {c:.4f}")

        # ASCII bar chart
        print(f"\nImpact profile across the sequence (each bar = group of {group_size}):")
        max_c = changes.max() + 1e-9
        bar_rows = 6
        for r in range(bar_rows, 0, -1):
            threshold = r / bar_rows
            line = "".join("█" if (c / max_c) >= threshold - 1/bar_rows else " " for c in changes)
            print(f"  {line}")
        print(f"  {'^' * num_groups}")
        print(f"  start                                                              end")


    # ── hyper-parameters ────────────────────────────────────────
    learning_rate  = 0.001
    num_epochs     = 50
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── GRU hyper-parameters ────────────────────────────────────
    hidden_size   = 16
    num_layers    = 1
    bidirectional = True
    dropout       = 0.1

    mean_criterion = nn.MSELoss()

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min((pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")

        # 3-way split: 1/2 train, 1/4 validate, 1/4 test
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)
        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)
        X_train_tensor    = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_tensor    = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_validate_tensor = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_validate_tensor = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_tensor     = torch.tensor(xtest.values,     dtype=torch.float32).to(device)
        Y_test_tensor     = torch.tensor(ytest.values,     dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader    = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),       batch_size=128, shuffle=True, generator=g)
        validate_loader = DataLoader(TensorDataset(X_validate_tensor, Y_validate_tensor), batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor),         batch_size=128, shuffle=False)
        input_size, output_size = X_train_tensor.shape[1], Y_train_tensor.shape[1]

        # ── build model ─────────────────────────────────────────
        mean_model = GRUMean(input_size, output_size,
                             hidden_size=hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=dropout).to(device)
        mean_optimizer = optim.Adam(mean_model.parameters(), lr=learning_rate)

        # ── train mean model ────────────────────────────────────
        best_val_loss, patience, no_improve = float("inf"), 5, 0
        for epoch in range(num_epochs):
            mean_model.train()
            for inputs, labels in train_loader:
                mean_optimizer.zero_grad()
                loss = mean_criterion(mean_model(inputs), labels)
                loss.backward()
                mean_optimizer.step()
            mean_model.eval()
            val_loss = sum(mean_criterion(mean_model(inp), lbl).item() for inp, lbl in validate_loader) / len(validate_loader)
            if val_loss < best_val_loss:
                best_val_loss, no_improve = val_loss, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        # ── DIAGNOSTICS: run after training ─────────────────────
        print("\n" + "#" * 72)
        print(f"#  GRU diagnostics for y_var={y_var}, node={node_num}, dt={dt}")
        print("#" * 72)
        sample = X_train_tensor[0]                    # shape (input_size,)
        diagnose_hidden_trajectories(mean_model, sample, top_k=5)
        diagnose_gate_values(mean_model, sample)
        diagnose_position_ablation(mean_model, sample, scaler_y=scaler_y, group_size=10)
        print("\n" + "#" * 72 + "\n")

        mean_preds, _ = predict_and_collect(mean_model, test_loader, mean_criterion, scaler_y)

        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])

        result = pd.concat([ytest.reset_index(), mean_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = valuation_models.groupby(["dt", "hr", "node_num"]).max().reset_index()
    keep_cols = ["dt", "hr", "node_num", "da_total_mean", "rt_total_mean"]
    direction_tag = "bi" if bidirectional else "uni"
    valuation_models["model"] = f"gru_{direction_tag}_{hidden_size}h_{num_layers}L_mean_only"

    return valuation_models

def GRU_framework_means_update_para(node_num, dt):

    run_number    = 1
    opexchange    = 'SPP'

    if int(run_number) == 1:
        data_location = 'training'
    else:
        data_location = 'secondRun'

    opexchange     = "SPP"
    data_location  = "training"
    gs_loc         = f"gs://ve_fourier/production/SPP/{data_location}"

    max_retries = 3
    retry_delay = 15
    for attempt in range(max_retries):
        try:
            data_df = pd.read_csv(f"{gs_loc}/{node_num}_{dt}.csv")
            data_df["dt"] = pd.to_datetime(data_df["dt"]).dt.strftime("%Y-%m-%d")
            break
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < max_retries - 1:
                time.sleep(retry_delay)
            else:
                raise

    selected_columns = [col for col in data_df.columns if "iirGen" not in col and "txoutage" not in col and "topGen" not in col]
    data_df = data_df[selected_columns]

    feb2021_dates = pd.DataFrame(pd.date_range("2021-02-13", "2021-02-19", freq="D"), columns=["dt"])["dt"].dt.strftime("%Y-%m-%d").tolist()
    if dt not in feb2021_dates:
        data_df = data_df[~data_df["dt"].isin(feb2021_dates)]

    if "dtHr" not in data_df.columns:
        data_df.insert(0, column="dtHr", value=pd.to_datetime(data_df["dt"]) + pd.to_timedelta(data_df["hr"] - 1, unit="h"))


    # ── model classes (all inside function) ─────────────────────

    class GRUMean(nn.Module):
        """
        GRU-based network for mean prediction.
        Treats the flat feature vector as a length-input_size sequence with 1 feature per step,
        runs a bidirectional multi-layer GRU, then projects the final hidden state to outputs.
        """
        def __init__(self, input_size, output_size, hidden_size=64, num_layers=2,
                     bidirectional=True, dropout=0.1):
            super(GRUMean, self).__init__()
            self.hidden_size   = hidden_size
            self.num_layers    = num_layers
            self.bidirectional = bidirectional
            self.gru = nn.GRU(
                input_size=1,                  # 1 value per sequence step
                hidden_size=hidden_size,
                num_layers=num_layers,
                batch_first=True,
                bidirectional=bidirectional,
                dropout=dropout if num_layers > 1 else 0.0,
            )
            # If bidirectional, the GRU outputs 2 * hidden_size features per step
            out_dim = hidden_size * (2 if bidirectional else 1)
            self.fc = nn.Linear(out_dim, output_size)

        def forward(self, x):
            # x shape: (B, input_size)
            x = x.unsqueeze(-1)                   # (B, input_size, 1) — seq_len=input_size, features=1
            out, h_n = self.gru(x)                # out: (B, input_size, hidden_size * num_directions)
            # Use the last time step's output (contains both directions if bidirectional)
            last = out.mean(dim=1)                 # (B, hidden_size * num_directions)
            return self.fc(last)                  # (B, output_size)


    def predict_and_collect(model, data_loader, criterion, scaler_y=None):
        model.eval()
        total_loss = 0.0
        predictions = []
        with torch.no_grad():
            for inputs, labels in data_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                total_loss += criterion(outputs, labels).item()
                predictions.append(outputs.cpu().numpy())
        predictions = np.concatenate(predictions, axis=0)
        if normalize and scaler_y is not None:
            predictions = scaler_y.inverse_transform(predictions)
        return predictions, total_loss / len(data_loader)


    # ── hyper-parameters ────────────────────────────────────────
    learning_rate  = 0.001
    num_epochs     = 50
    normalize      = True
    device         = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── GRU hyper-parameters ────────────────────────────────────
    hidden_size   = 16
    num_layers    = 1
    bidirectional = True
    dropout       = 0.1

    mean_criterion = nn.MSELoss()

    valuation_models = pd.DataFrame()

    for y_var in [["da_total", "rt_total"]]:
        trainingPeriod = int(min((pd.to_datetime(dt) - pd.to_datetime("2017-01-01")) / np.timedelta64(1, "D"), 730))
        xtrain, ytrain, xtest, ytest = ve_model_functions.getTrainTestData(
            opexchange, data_df, y=y_var, bidDt=dt, hour_gap=18,
            trainingPeriod=str(trainingPeriod) + "D", train_a_or_f="f")

        # 3-way split: 1/2 train, 1/4 validate, 1/4 test
        xtrain, xtest, ytrain, ytest = train_test_split(xtrain, ytrain, test_size=1/8, shuffle=False)
        xtrain, xvalidate, ytrain, yvalidate = train_test_split(xtrain, ytrain, test_size=1/7, shuffle=False)
        scaler_x, scaler_y = None, None
        if normalize:
            scaler_x = StandardScaler()
            scaler_y = StandardScaler()
            xtrain    = pd.DataFrame(scaler_x.fit_transform(xtrain),    columns=xtrain.columns,    index=xtrain.index)
            xvalidate = pd.DataFrame(scaler_x.transform(xvalidate),     columns=xvalidate.columns, index=xvalidate.index)
            xtest     = pd.DataFrame(scaler_x.transform(xtest),         columns=xtest.columns,     index=xtest.index)
            ytrain    = pd.DataFrame(scaler_y.fit_transform(ytrain),    columns=ytrain.columns,    index=ytrain.index)
            yvalidate = pd.DataFrame(scaler_y.transform(yvalidate),     columns=yvalidate.columns, index=yvalidate.index)
            ytest     = pd.DataFrame(scaler_y.transform(ytest),         columns=ytest.columns,     index=ytest.index)
        X_train_tensor    = torch.tensor(xtrain.values,    dtype=torch.float32).to(device)
        Y_train_tensor    = torch.tensor(ytrain.values,    dtype=torch.float32).to(device)
        X_validate_tensor = torch.tensor(xvalidate.values, dtype=torch.float32).to(device)
        Y_validate_tensor = torch.tensor(yvalidate.values, dtype=torch.float32).to(device)
        X_test_tensor     = torch.tensor(xtest.values,     dtype=torch.float32).to(device)
        Y_test_tensor     = torch.tensor(ytest.values,     dtype=torch.float32).to(device)

        g = torch.Generator()
        g.manual_seed(42)
        train_loader    = DataLoader(TensorDataset(X_train_tensor, Y_train_tensor),       batch_size=128, shuffle=True, generator=g)
        validate_loader = DataLoader(TensorDataset(X_validate_tensor, Y_validate_tensor), batch_size=128, shuffle=False)
        test_loader     = DataLoader(TensorDataset(X_test_tensor, Y_test_tensor),         batch_size=128, shuffle=False)
        input_size, output_size = X_train_tensor.shape[1], Y_train_tensor.shape[1]

        # ── build model ─────────────────────────────────────────
        mean_model = GRUMean(input_size, output_size,
                             hidden_size=hidden_size, num_layers=num_layers,
                             bidirectional=bidirectional, dropout=dropout).to(device)
        mean_optimizer = optim.Adam(mean_model.parameters(), lr=learning_rate)

        # ── train mean model ────────────────────────────────────
        best_val_loss, patience, no_improve = float("inf"), 5, 0
        for epoch in range(num_epochs):
            mean_model.train()
            for inputs, labels in train_loader:
                mean_optimizer.zero_grad()
                loss = mean_criterion(mean_model(inputs), labels)
                loss.backward()
                mean_optimizer.step()
            mean_model.eval()
            val_loss = sum(mean_criterion(mean_model(inp), lbl).item() for inp, lbl in validate_loader) / len(validate_loader)
            if val_loss < best_val_loss:
                best_val_loss, no_improve = val_loss, 0
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"Early stopping at epoch {epoch + 1}")
                    break

        mean_preds, _ = predict_and_collect(mean_model, test_loader, mean_criterion, scaler_y)

        if normalize and scaler_y is not None:
            ytest = pd.DataFrame(scaler_y.inverse_transform(ytest), columns=ytest.columns, index=ytest.index)

        mean_df = pd.DataFrame(mean_preds, columns=[f"{y}_mean" for y in y_var])

        result = pd.concat([ytest.reset_index(), mean_df], axis=1)
        result["dtHr"] = pd.to_datetime(result["dtHr"])
        result["dt"]   = result["dtHr"].dt.strftime("%Y-%m-%d")
        result["hr"]   = result["dtHr"].dt.hour + 1
        valuation_models = pd.concat([valuation_models, result])

    valuation_models = valuation_models.groupby(["dt", "hr", "node_num"]).max().reset_index()
    keep_cols = ["dt", "hr", "node_num", "da_total_mean", "rt_total_mean"]
    direction_tag = "bi" if bidirectional else "uni"
    valuation_models["model"] = f"gru_{direction_tag}_{hidden_size}h_{num_layers}L_mean_only"

    return valuation_models
    
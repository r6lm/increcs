# load model for test
model = get_model(model_params)
try:
    model.load_state_dict(torch.load(model_checkpoint_path))
    print("loaded model:", model_checkpoint_path)
except FileNotFoundError:
    latest_checkpoint_path = get_path_from_re(
        f'{model_checkpoint_subdir}/'f'{train_params["model_filename_stem"]}*.pth')
    model.load_state_dict(torch.load(latest_checkpoint_path))
    print("loaded model:", latest_checkpoint_path)

class Path_Hyperparameter:
    random_seed = 42

    # Dataset and path defaults. Server launchers override the root and folder names.
    dataset_name = '/home/xxj/Data/LEVIR_CD'
    image_dir_a = 'A'
    image_dir_b = 'B'
    label_dir = 'label'

    # Training defaults recovered from the repository history.
    epochs = 200
    batch_size = 8
    inference_ratio = 2
    learning_rate = 2e-4
    factor = 0.1
    patience = 12
    warm_up_step = 500
    weight_decay = 1e-3
    amp = True
    load = False
    max_norm = 20

    # Evaluation and checkpoint defaults.
    evaluate_epoch = 100
    stage_epoch = [0, 0, 0, 0, 0]
    save_checkpoint = True
    save_interval = 10
    save_best_model = True
    log_wandb_project = 'creloss'

    # Data and model defaults.
    noise_p = 0.3
    dropout_p = 0.1
    patch_size = 256
    y = 2
    b = 1
    log_path = './log_feature/'

    def state_dict(self):
        values = {
            key: value for key, value in vars(type(self)).items()
            if not key.startswith('_') and not callable(value)
        }
        values.update(vars(self))
        return values


ph = Path_Hyperparameter()

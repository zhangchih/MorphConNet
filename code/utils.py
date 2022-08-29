import os
from shutil import copyfile


def _create_model_training_folder(log_dir, files_to_same):
    model_checkpoints_folder = os.path.join(log_dir, 'checkpoints')
    if not os.path.exists(model_checkpoints_folder):
        os.makedirs(model_checkpoints_folder)
        for file in files_to_same:
            copyfile(file, os.path.join(model_checkpoints_folder, os.path.basename(file)))
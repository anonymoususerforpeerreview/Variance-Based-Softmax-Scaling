import os

import numpy as np
import wandb

from shared.utils import get_train_id, get_analysis_id
from uncertainty.analysis.visual import Visual


class W:  # W for wandb

    @staticmethod
    def get_wandb_run_train(h: dict):
        return get_train_id(h)

    @staticmethod
    def get_wandb_run_analysis(h: dict):
        return get_analysis_id(h)

    @staticmethod
    def get_wandb_project(h: dict, is_train: bool = True):
        assert h['method'] in ['UQ_rednd', 'ensemble', 'mc_dropout'], "Method not supported for wandb project name."

        project_name = f"{h['method']}_{h['dataset']}_{h['wandb_project']}"
        entity = h['wandb_entity']
        if is_train:
            run_name = W.get_wandb_run_train(h)
        else: # analysis
            run_name = W.get_wandb_run_analysis(h)
        # replace empty string with None
        entity = entity if entity != "" else None
        return entity, project_name, run_name

    @staticmethod
    def initialize_wandb(h: dict, entity, project_name, run_name, write_to_disk: bool):
        # conf = vars(h)
        conf = h.copy()
        # store conf._classifier_configs them as string instead of type Dict[ClassifierKey, ...]
        # Fixes ClassifierKey not serializable error for wandb
        # conf["_classifier_configs"] = str(conf["_classifier_configs"])

        wandb.init(entity=entity, project=project_name, name=run_name, config=conf)

        # After initializing the wandb run, get the run id
        run_id = wandb.run.id
        if write_to_disk:
            # Save the run id to a file in the logs directory
            if os.path.exists(os.path.join(h['log_path'], 'wandb_run_id.txt')):
                os.remove(os.path.join(h['log_path'], 'wandb_run_id.txt'))
            with open(os.path.join(h['log_path'], 'wandb_run_id.txt'), 'w') as f:
                f.write(run_id)
                # write project name to file
                f.write(f"\n{project_name}")

        return run_id

    @staticmethod
    def retrieve_existing_wandb_run_id(h: dict):
        # Save the run id to a file in the logs directory
        if os.path.exists(os.path.join(h['log_path'], 'wandb_run_id.txt')):
            with open(os.path.join(h['log_path'], 'wandb_run_id.txt'), 'r') as f:
                text = f.read()
                # first line is the run id, second line is the project name (second line is optional)
                run_id = text.split('\n')[0]
                project_name = text.split('\n')[1] if len(text.split('\n')) > 1 else None

        # if file doesn't exist, return None
        else:
            run_id = None
            project_name = None

        assert run_id is not None, "Run id not found, set use_wandb to False in the config file to disable wandb logging"
        assert project_name is not None, "Project name not found, set use_wandb to False in the config file to disable wandb logging"

        return run_id, project_name

    @staticmethod
    def log_im(h: dict, im, name: str, show_locally: bool = False):
        if h['use_wandb']:
            wandb.log({name: [wandb.Image(im)]})
        elif show_locally:
            Visual.plot_img(im, name)

    @staticmethod
    def log_ims(h: dict, ims: list, name: str, show_locally: bool = False):
        for i, im in enumerate(ims):
            W.log_im(h, im, f"{name}_{i}", show_locally)

    @staticmethod
    def log_x_y(h: dict, x_values: np.ndarray, y_values: np.ndarray, name: str, x_label: str, y_label: str,
                show_locally: bool = False):
        if h['use_wandb']:
            data = [[x, y] for (x, y) in zip(x_values, y_values)]
            table = wandb.Table(data=data, columns=[x_label, y_label])
            wandb.log(
                {
                    name: wandb.plot.line(
                        table, x_label, y_label, title=name
                    )
                }
            )
        elif show_locally:
            # display the plot locally
            Visual.plot_x_y_scatter(x_values, y_values, name, x_label, y_label)

    @staticmethod
    def log_scalar(h: dict, name: str, value: float):
        if h['use_wandb']:
            wandb.log({name: value})
        else:
            print(f"{name}: {value}")

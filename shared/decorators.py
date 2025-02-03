import time

import wandb
import torch
import gc

from shared.utils import set_seed, create_path_if_not_exists
from shared.wandb import W  # wandb wrapper


def timer_decorator(func):
    def wrapper(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        print(f"Execution time of {func.__name__}: {end_time - start_time} seconds")
        return result

    return wrapper


def wandb_decorator(func):
    """
    Initialize wandb if h['use_wandb'] is True, and finish the run after the function is done.
    Warning: ensure that the first argument of the function with this decorator is the h object!
    """

    def wrapper(h: dict, *args, **kwargs):
        assert type(h) == dict, \
            ("First argument must be an dict object. "
             "When using this decorator, the first argument must be the h object."
             "eg: @wandb_decorator\n"
             "def _main(h: dict):"
             "    ...")

        if h['use_wandb']:
            entity, project_name, run_name = W.get_wandb_project(h, is_train=True)
            print(f"entity: {entity}, project_name: {project_name}, run_name: {run_name}")
            run_id = W.initialize_wandb(h, entity, project_name, run_name, write_to_disk=True)

            wandb.config.update({"initial_training_run_id": run_id})  # store the run id of the initial training run
            wandb.run.tags = wandb.run.tags + (f"train_run_id={run_id}",)  # for easy filtering in the wandb dashboard

        result = func(h, *args, **kwargs)

        if h['use_wandb']:
            wandb.finish()
        return result

    return wrapper


def wandb_resume_decorator(func):
    def wrapper(h: dict, *args, **kwargs):
        assert type(h) == dict, \
            ("See `wandb_decorator` for more information.")

        if h['use_wandb']:
            run_id, project_name = W.retrieve_existing_wandb_run_id(h)
            wandb.init(id=run_id, resume="allow", project=project_name, entity=h['wandb_entity'])

        result = func(h, *args, **kwargs)

        if h['use_wandb']:
            wandb.finish()

        return result

    return wrapper


def wandb_resume_but_new_run_decorator(func):
    def wrapper(h: dict, *args, **kwargs):
        assert type(h) == dict, \
            ("See `wandb_decorator` for more information.")

        if h['use_wandb']:
            if not h['wandb_analysis_independent_of_train_run']:  # Typical case: run based on existing training run
                run_id, project_name = W.retrieve_existing_wandb_run_id(h)
                run_name = W.get_wandb_run_analysis(h)
                W.initialize_wandb(h, h['wandb_entity'], project_name, run_name, write_to_disk=False)

                wandb.config.update({"initial_training_run_id": run_id})  # store the run id of the initial training run
                wandb.run.tags = wandb.run.tags + (
                    f"train_run_id={run_id}",)  # for easy filtering in the wandb dashboard

            else:  # Entirely new run, independent of previous runs (eg a train run)
                entity, project_name, run_name = W.get_wandb_project(h, is_train=False)
                print(f"entity: {entity}, project_name: {project_name}, run_name: {run_name}")
                run_id = W.initialize_wandb(h, entity, project_name, run_name, write_to_disk=False)
        else:
            print("Wandb is disabled. Not resuming any runs.")

        result = func(h, *args, **kwargs)

        if h['use_wandb']:
            wandb.finish()

        return result

    return wrapper


def init_decorator(func):
    def init(h: dict):
        torch.set_float32_matmul_precision('medium')
        torch.cuda.empty_cache()
        gc.collect()

        # set random seeds
        set_seed(h['seed'])

        create_path_if_not_exists(h['dataset_path'])
        create_path_if_not_exists(h['log_path'])

    def wrapper(h: dict, *args, **kwargs):
        assert type(h) == dict, \
            ("See `wandb_decorator` for more information.")

        init(h)
        result = func(h, *args, **kwargs)

        return result

    return wrapper

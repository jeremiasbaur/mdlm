
import os
from collections import defaultdict
import argparse
import pickle

import hydra
import lightning as L
from hydra.core.global_hydra import GlobalHydra
from omegaconf import OmegaConf, open_dict
import torch
import numpy as np

import dataloader
import diffusion
import utils

OmegaConf.register_new_resolver(
  'cwd', os.getcwd)
OmegaConf.register_new_resolver(
  'device_count', torch.cuda.device_count)
OmegaConf.register_new_resolver(
  'eval', eval)
OmegaConf.register_new_resolver(
  'div_up', lambda x, y: (x + y - 1) // y)

def create_dict(path: str, model_family: str, model_name: str):
  """
  creates or retrieves a dictionary for a specified model family and model name,
  storing or retrieving it in a specific directory structure

  parameters:
  model_family: string representing the family of the model (e.g., 'transformer', 'bert')
  model_name: string representing the specific name of the model

  returns:
  tuple containing:
  - models: dictionary loaded from the file if it exists, or an empty dictionary if the file does not exist
  - dir: string path to the directory where the model dictionary is stored
  """

  os.makedirs(os.path.join(path, model_family), exist_ok = True)
  
  dir = os.path.join(path, model_family, model_name)
  if os.path.isfile(dir):
    with open(dir, 'rb') as file:
      models = pickle.load(file)
  else:
    models = {}

  return models, dir


def load_config(model_dir, model_ckpt, eval_dataset, repo_config_path="./configs", repo_config_name="config"):
  """
  Reconstructs the training config by merging original overrides with validation settings.
  """

  overrides_path = os.path.abspath(os.path.join(model_dir, ".hydra", "overrides.yaml"))
  full_ckpt_path = os.path.abspath(os.path.join(model_dir, 'checkpoints', model_ckpt))

  if not os.path.exists(overrides_path):
    raise FileNotFoundError(f"Could not find overrides at {overrides_path}")

  original_overrides = list(OmegaConf.load(overrides_path))

  val_overrides = [
      "mode=ppl_eval",
      f"data={eval_dataset}",
      "loader.eval_batch_size=16", 
      "+wandb.offline=true",
      "eval.compute_generative_perplexity=True",
      f"eval.checkpoint_path={full_ckpt_path}",
  ]

  final_overrides = original_overrides + val_overrides

  GlobalHydra.instance().clear()
  with hydra.initialize(version_base=None, config_path=repo_config_path):
    config = hydra.compose(config_name=repo_config_name, overrides=final_overrides)
      
  return config

def ppl_eval(config, logger, tokenizer):
  logger.info('Starting Zero Shot Eval.')
  model = diffusion.Diffusion.load_from_checkpoint(
              config.eval.checkpoint_path,
              tokenizer=tokenizer,
              config=config)

  if config.eval.disable_ema:
    logger.info('Disabling EMA.')
    model.ema = None

  callbacks = []
  if 'callbacks' in config:
    for _, callback in config.callbacks.items():
      callbacks.append(hydra.utils.instantiate(callback))

  trainer = hydra.utils.instantiate(
      config.trainer,
      default_root_dir=os.getcwd(),
      callbacks=callbacks,
      strategy=hydra.utils.instantiate(config.strategy),
      logger=None,)
  _, valid_ds = dataloader.get_dataloaders(
      config, tokenizer, skip_train=True, valid_seed=config.seed)
  return trainer.validate(model, valid_ds)


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_dir", type=str, required=True, 
                        help="Path to the model directory (containing .hydra and checkpoints)")
  parser.add_argument("--ckpt", type=str, required=True, 
                      help="Name of the checkpoint file (e.g. '5-102000.ckpt')")
  parser.add_argument("--config_path", type=str, default="./configs",
                      help="Path to the source code configs directory")
  parser.add_argument("--eval_dataset", type=str, default="wikitext2")
  parser.add_argument("--save_dir", type=str, required=True)

  args = parser.parse_args()
  config = load_config(args.model_dir, args.ckpt, args.eval_dataset, repo_config_path=args.config_path)

  logger = utils.get_logger(__name__)
  tokenizer = dataloader.get_tokenizer(config)

  print(f"Validating checkpoint: {config.eval.checkpoint_path}")
  print(f"\nModel name: {config.wandb.name}\n")
  runs = []
  for i in range(5):
    L.seed_everything(config.seed+1+i)
    runs.extend(ppl_eval(config, logger, tokenizer))

  metrics_dict = defaultdict(list)
  for run in runs:
    for key, value in run.items():
      metrics_dict[key].append(value)
  
  final_stats = {}
  print(f'{args.eval_dataset}: {config.wandb.name} at {args.ckpt}')

  for key, values in metrics_dict.items():
    # Convert the list to a numpy array
    arr = np.array(values)
    final_stats[key] = arr
    
    # Calculate Mean and Std Deviation
    mean_val = np.mean(arr)
    std_val = np.std(arr)

    print(f"{key:<20} | {mean_val:.4f}     | {std_val:.4f}")
    
  filename = f'{config.wandb.name}.pkl'
  
  models, dir = create_dict(args.save_dir, '', filename)
  global_steps = int(args.ckpt.split('-')[1].split('.')[0])
  
  if args.eval_dataset in models:
    models[args.eval_dataset][global_steps] = final_stats 
  else:
    models[args.eval_dataset] = {global_steps: final_stats}

  with open(dir, 'wb') as file:
    pickle.dump(models, file)

  print(f"saved pickle to {dir}")

if __name__ == '__main__':
  main()

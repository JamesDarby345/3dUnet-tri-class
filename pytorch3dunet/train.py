import random
import wandb
import torch
import json

from pytorch3dunet.unet3d.config import load_config, copy_config
from pytorch3dunet.unet3d.trainer import create_trainer
from pytorch3dunet.unet3d.utils import get_logger

logger = get_logger('TrainingSetup')


def main():
    # Load and log experiment configuration
    config, config_path = load_config()
    logger.info(config)

    train_file_paths = config['loaders']['train']['file_paths']
    patch_shape = config['loaders']['train']['slice_builder']['patch_shape']
    transforms = config['loaders']['train']['transformer']

    # Initialize Weights and Biases
    wandb.init(project='tri-class-3dunet-vesuvius', config=config)

    wandb.config.update({
        'transformers': config['loaders']['train'].get('transformer', {}),
        'patch_shape': config['loaders']['train']['slice_builder'].get('patch_shape', []),
        'train_file_paths': config['loaders']['train'].get('file_paths', []),
    }, allow_val_change=True)

    config_filename = 'config.json'
    with open(config_filename, 'w') as f:
        json.dump(config, f, indent=4)

    config_artifact = wandb.Artifact('experiment-config', type='config')
    config_artifact.add_file(config_filename)
    wandb.log_artifact(config_artifact)

    manual_seed = config.get('model').get('manual_seed', None)

    if manual_seed is not None:
        print(f'Seed the RNG for all devices with {manual_seed}')
        logger.info(f'Seed the RNG for all devices with {manual_seed}')
        logger.warning('Using CuDNN deterministic setting. This may slow down the training!')
        random.seed(manual_seed)
        torch.manual_seed(manual_seed)
        # see https://pytorch.org/docs/stable/notes/randomness.html
        torch.backends.cudnn.deterministic = True

    # Create trainer
    trainer = create_trainer(config)
    # Copy config file
    copy_config(config, config_path)
    # Start training
    trainer.fit()

    wandb.finish()


if __name__ == '__main__':
    main()

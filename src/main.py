"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 12, 2025
 * Purpose: Main entry point for the 3D-VAE-GAN project. Handles command-line arguments,
            loads configuration from YAML files, and runs the appropriate training
            or testing function based on specified parameters.
"""
import argparse
import yaml
# Import training functions
from train import train_vae, train_multiview, train_vae_pose, train_multiview_pose


def str2bool(v: str) -> bool:
    """Convert string to boolean"""
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def main():
    """Main entry point"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='3D-VAEGAN for 3D shape generation')
    parser.add_argument('--config', type=str, default='../config/default.yaml', help='Path to config file')
    parser.add_argument('--alg_type', type=str,
                        choices=['3DVAEGAN', '3DVAEGAN_MULTIVIEW', '3DVAEGAN_POSE', '3DVAEGAN_MULTIVIEW_POSE'],
                        help='Algorithm type to use')
    parser.add_argument('--batch_size', type=int, help='Batch size (overrides config)')
    parser.add_argument('--obj', type=str, help='Object category (overrides config)')
    parser.add_argument('--mode', type=str, choices=['train', 'test'], help='Run mode')
    parser.add_argument('--n_epochs',  help='Number of epochs (overrides config)')
    args, unknown = parser.parse_known_args()

    # Load configuration
    with open(args.config, 'r') as f:
        config = yaml.safe_load(f)

    # Override with command line arguments
    for arg in vars(args):
        if getattr(args, arg) is not None and arg != 'config':
            config[arg] = getattr(args, arg)

    # Create args namespace with config values
    config_args = argparse.Namespace(**config)

    # Dictionary of training/testing functions
    functions = {
        'train': {
            '3DVAEGAN': train_vae,
            '3DVAEGAN_MULTIVIEW': train_multiview,
            '3DVAEGAN_POSE': train_vae_pose,
            '3DVAEGAN_MULTIVIEW_POSE': train_multiview_pose
        },
        'test': {
            # To be implemented
        }
    }

    # Run appropriate function
    mode = config.get('mode', 'train')
    alg_type = config.get('alg_type', '3DVAEGAN')

    print(f"Running {alg_type} in {mode} mode")
    if mode in functions and alg_type in functions[mode]:
        functions[mode][alg_type](config_args)
    else:
        print(f"Error: {alg_type} is not implemented for {mode} mode")


if __name__ == '__main__':
    main()
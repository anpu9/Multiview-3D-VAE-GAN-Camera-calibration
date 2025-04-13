"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 2025
 * Purpose: Implements the training procedures for the 3D-VAE-GAN model.
            Supports training in single-view, multi-view, and pose-aware modes,
            with comprehensive logging, checkpoint saving and visualization of the training progress.
"""
import torch
from torch import optim
from torch import nn
from collections import OrderedDict
import os
from utils import make_hyparam_string, save_new_pickle, read_pickle, SavePloat_Voxels, generateZ
from utils import CO3DDataset, var_or_cuda
from model import Discriminator, MultiViewEncoder, SingleViewEncoder, Generator
from lr_sh import MultiStepLR
torch.autograd.set_detect_anomaly(True)

def KLLoss(z_mu, z_var):
    # Non-in-place version
    return -0.5 * torch.sum(1 + z_var - z_mu.pow(2) - torch.exp(z_var))

def train_model(args, multiview=False, use_pose=False):
    """
    Generic training function for 3D-VAEGAN models that handles both single and multi-view configurations.

    Args:
        args: Command line arguments
        multiview: Whether to use multiple views per object
        use_pose: Whether to incorporate pose information
    """
    # Set up hyperparameter logging
    hyper_params = [
        ("model", args.model_name),
        ("cube", args.cube_len),
        ("bs", args.batch_size),
        ("g_lr", args.g_lr),
        ("d_lr", args.d_lr),
        ("z", args.z_dis),
        ("bias", args.bias),
        ("sl", args.soft_label),
    ]

    # Add multiview-specific parameters if needed
    if multiview:
        hyper_params.extend([
            ("views", args.num_views),
            ("combine", args.combine_type),
        ])

    # Add pose-specific parameters if needed
    if use_pose:
        hyper_params.append(("pose", "enabled"))

    hyparam_dict = OrderedDict(((arg, value) for arg, value in hyper_params))
    log_param = make_hyparam_string(hyparam_dict)
    print(f"Training {'multiview' if multiview else 'single-view'} model with hyperparameters: {log_param}")

    # Setup TensorBoard
    summary_writer = None
    if args.use_tensorboard:
        try:
            from torch.utils.tensorboard import SummaryWriter
            summary_writer = SummaryWriter(os.path.join(args.output_dir, args.log_dir, log_param))

            def inject_summary(summary_writer, tag, value, step):
                summary_writer.add_scalar(tag, value, step)
        except ImportError:
            print("tensorboard package not found. Disabling TensorBoard logging.")
            args.use_tensorboard = False

    # Load dataset
    train_dataset = CO3DDataset(
        root=args.input_dir,
        args=args,
        multiview=multiview,
        use_pose=use_pose,
        apply_mask=True
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True
    )

    # Initialize models
    D = Discriminator(args)  # or _D for backward compatibility
    G = Generator(args)  # or _G for backward compatibility

    # Initialize encoder based on configuration
    if multiview:
        E = MultiViewEncoder(args)  # or _E_MultiView for backward compatibility
    else:
        E = SingleViewEncoder(args)  # or _E for backward compatibility

    # Initialize optimizers
    D_solver = optim.Adam(D.parameters(), lr=args.d_lr, betas=args.beta)
    G_solver = optim.Adam(G.parameters(), lr=args.g_lr, betas=args.beta)
    E_solver = optim.Adam(E.parameters(), lr=args.e_lr, betas=args.beta)

    # Setup learning rate schedulers
    schedulers = {}
    if args.lrsh:
        schedulers['D'] = MultiStepLR(D_solver, milestones=[500, 1000])
        schedulers['G'] = MultiStepLR(G_solver, milestones=[500, 1000])
        schedulers['E'] = MultiStepLR(E_solver, milestones=[500, 1000])

    # Move models to GPU if available
    if torch.cuda.is_available():
        print("Using CUDA")
        D.cuda()
        G.cuda()
        E.cuda()

    # Loss function
    criterion = nn.BCELoss()

    # Try to load previous checkpoint
    pickle_path = os.path.join(args.output_dir, args.pickle_dir, log_param)
    try:
        read_pickle(pickle_path, G, G_solver, D, D_solver, E, E_solver)
        load_success = True
    except Exception as e:
        print(f"Failed to load models: {e}")
        print("Starting training from scratch")
    if load_success:
        print(f"Loaded models from {pickle_path}")

    # Main training loop
    for epoch in range(args.n_epochs):
        for batch_idx, batch_data in enumerate(train_loader):
            # Handle different data formats for single/multi-view
            if multiview:
                images, model_3d = batch_data
                # Skip incomplete batches
                if len(images) < args.num_views or any(img.size(0) != args.batch_size for img in images):
                    continue
            else:
                image, model_3d = batch_data

                # Skip incomplete batches
                if image.size(0) != args.batch_size:
                    continue

            # Move 3D model data to GPU
            model_3d = var_or_cuda(model_3d)

            # Generate random latent vectors for GAN
            Z = generateZ(args)

            # Process through the encoder (single or multi-view)
            if multiview:
                Z_vae, z_mus, z_vars = E(images)
            else:
                z_mu, z_var = E(image)
                Z_vae = E.reparameterize(z_mu, z_var)

            # Generate 3D model from encoded representation
            G_vae = G(Z_vae)

            # Create labels for discriminator (with optional soft labels)
            if args.soft_label:
                real_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0.7, 1.0))
                fake_labels = var_or_cuda(torch.Tensor(args.batch_size).uniform_(0, 0.3))
            else:
                real_labels = var_or_cuda(torch.ones(args.batch_size))
                fake_labels = var_or_cuda(torch.zeros(args.batch_size))

            # ============= Train the discriminator =============
            # Real samples
            d_real = D(model_3d).view_as(real_labels)
            d_real_loss = criterion(d_real, real_labels)

            # Fake samples
            fake = G(Z)
            d_fake = D(fake.detach()).view_as(fake_labels)
            d_fake_loss = criterion(d_fake, fake_labels)

            d_loss = d_real_loss + d_fake_loss

            # Calculate discriminator accuracy
            d_real_acu = torch.ge(d_real, 0.5).float()
            d_fake_acu = torch.le(d_fake, 0.5).float()
            d_total_acu = torch.mean(torch.cat((d_real_acu, d_fake_acu), 0))

            # Only update discriminator if it's not too strong
            if d_total_acu <= args.d_thresh:
                D.zero_grad()
                d_loss.backward()
                D_solver.step()

            # ============= Train the encoder =============
            # Compute reconstruction loss
            recon_loss_e = torch.sum(torch.pow((G_vae - model_3d), 2))

            # Compute KL divergence loss (different handling for single/multi-view)
            if multiview:
                kl_loss = 0
                for i in range(args.num_views):
                    kl_loss = kl_loss + KLLoss(z_mus[i], z_vars[i])
                kl_loss = kl_loss / args.num_views
            else:
                kl_loss = KLLoss(z_mu, z_var)

            # Total encoder loss
            E_loss = recon_loss_e + kl_loss

            E.zero_grad()
            E_loss.backward()
            E_solver.step()

            # ============= Train the generator =============
            # Generate new fake samples for discriminator path
            Z_new = generateZ(args)
            fake_new = G(Z_new)
            d_fake_new = D(fake_new).view_as(real_labels)
            g_gan_loss = criterion(d_fake_new, real_labels)

            Z_vae_detached = Z_vae.detach()  # Detach to prevent gradients flowing to encoder
            G_vae_for_gen = G(Z_vae_detached)
            recon_loss_g = torch.sum(torch.pow((G_vae_for_gen - model_3d), 2))

            # Combine losses
            g_loss = g_gan_loss + recon_loss_g

            G.zero_grad()
            g_loss.backward()
            G_solver.step()
        # =============== Log progress after each epoch ===============
        iteration = epoch + 1
        print(f'Epoch {iteration}/{args.n_epochs}; '
              f'D_loss: {d_loss.item():.4f} (R: {d_real_loss.item():.4f}, F: {d_fake_loss.item():.4f}), '
              f'G_loss: {g_loss.item():.4f}, Recon: {recon_loss_g.item():.4f}), '
              f'E_loss: {E_loss.item():.4f} (Recon: {recon_loss_e.item():.4f}, KL: {kl_loss.item():.4f}), '
              f'D_acc: {d_total_acu.item():.4f}')

        # Log to TensorBoard
        if args.use_tensorboard and summary_writer is not None:
            log_save_path = os.path.join(args.output_dir, args.log_dir, log_param)
            os.makedirs(log_save_path, exist_ok=True)

            info = {
                # Discriminator losses
                'loss/D/total': d_loss.item(),
                'loss/D/real': d_real_loss.item(),
                'loss/D/fake': d_fake_loss.item(),
                'loss/D/accuracy': d_total_acu.item(),

                # Generator losses
                'loss/G/total': g_loss.item(),
                'loss/G/reconstruction': recon_loss_g.item(),

                # Encoder losses
                'loss/E/total': E_loss.item(),
                'loss/E/reconstruction': recon_loss_e.item(),
                'loss/E/kl': kl_loss.item()
            }

            for tag, value in info.items():
                inject_summary(summary_writer, tag, value, iteration)

            summary_writer.flush()

            # You can also add visualizations of your 3D models to TensorBoard
            if (epoch + 1) % args.image_save_step == 0:
                # Convert one sample to a 3D grid for visualization
                sample = fake[0].detach().cpu().numpy().squeeze()
                summary_writer.add_histogram('voxel_values', sample, iteration)
        # =============== Save sample images periodically ===============
        if (epoch + 1) % args.image_save_step == 0:
            samples = fake.cpu().data[:8].numpy()
            image_path = os.path.join(args.output_dir, args.image_dir, log_param)
            os.makedirs(image_path, exist_ok=True)
            SavePloat_Voxels(samples, image_path, iteration)

        # =============== Save model checkpoint periodically ===============
        if (epoch + 1) % args.pickle_step == 0:
            pickle_save_path = os.path.join(args.output_dir, args.pickle_dir, log_param)
            os.makedirs(pickle_save_path, exist_ok=True)
            save_new_pickle(pickle_save_path, iteration, G, G_solver, D, D_solver, E, E_solver)

        # =============== Update learning rate schedulers ===============
        if args.lrsh:
            try:
                for scheduler in schedulers.values():
                    scheduler.step()
            except Exception as e:
                print(f"Failed to update learning rate schedulers: {e}")
    if summary_writer is not None:
        summary_writer.close()

# Wrapper functions for specific configurations
def train_vae(args):
    """Train single-view model without pose information"""
    train_model(args, multiview=False, use_pose=False)

def train_multiview(args):
    """Train multi-view model without pose information"""
    train_model(args, multiview=True, use_pose=False)

def train_vae_pose(args):
    """Train single-view model with pose information"""
    train_model(args, multiview=False, use_pose=True)
def train_multiview_pose(args):
    """Train multi-view model with pose information"""
    train_model(args, multiview=True, use_pose=True)

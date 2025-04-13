"""
 * Authors: Yuyang Tian and Arun Mekkad
 * Date: April 12, 2025
 * Purpose: Defines the neural network architectures for the 3D-VAE-GAN model.
            Includes classes for the Generator, Discriminator, and various Encoder
            variants (single-view and multi-view), with support for different
            configuration options.
 TODO: Define posing encoding
"""
import torch
import math
from utils import var_or_cuda
torch.autograd.set_detect_anomaly(True)

# Base class for all models with common utilities
class ModelBase(torch.nn.Module):
    def __init__(self, args):
        super(ModelBase, self).__init__()
        self.args = args

    def _calculate_output_size(self, size, kernel_size, stride, padding, dilation=1):
        """Calculate output size for a convolutional layer"""
        return ((size + 2 * padding - dilation * (kernel_size - 1) - 1) // stride) + 1

    def _calculate_output_size_transpose(self, size, kernel_size, stride, padding, output_padding=0, dilation=1):
        """Calculate output size for a transposed convolutional layer"""
        return (size - 1) * stride - 2 * padding + dilation * (kernel_size - 1) + output_padding + 1

    def _get_padding(self, in_size, out_size, kernel_size, stride, dilation=1):
        """Calculate padding needed to achieve a specific output size"""
        padding = math.ceil((1 / 2) * (dilation * (kernel_size - 1) + (out_size - 1) * stride + (1 - in_size)))
        return padding

    def _get_valid_padding(self, size, kernel_size, stride):
        """Calculate padding for valid convolution"""
        padding = math.ceil(float(size - kernel_size + 1) / float(stride))
        return padding

    def output_features(self, size, kernel_size, stride, padding):
        """Calculate output feature map size"""
        out = (((size - kernel_size) + (2 * padding)) // stride) + 1
        return out

# Generator: transform a latent vector into a 3D voxel grid representation
class Generator(ModelBase):
    def __init__(self, args):
        super(Generator, self).__init__(args)
        self.cube_len = args.cube_len
        self.z_size = args.z_size

        # Determine padding based on cube length
        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1, 1, 1)

        # Create transposed convolutional layers
        self.layers = self._build_layers(padd)

    def _build_layers(self, padd):
        """Build the generator layers"""
        return torch.nn.ModuleList([
            # Layer 1: [batch_size, z_size, 1, 1, 1] -> [batch_size, cube_len*8, 4, 4, 4]
            torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.z_size, self.cube_len * 8, kernel_size=4, stride=2, bias=self.args.bias,padding=padd),
                torch.nn.BatchNorm3d(self.cube_len * 8),
                torch.nn.ReLU()
            ),
            # Layer 2: [batch_size, cube_len*8, 4, 4, 4] -> [batch_size, cube_len*4, 8, 8, 8]
            torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.cube_len * 8, self.cube_len * 4, kernel_size=4, stride=2, bias=self.args.bias, padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len * 4),
                torch.nn.ReLU()
            ),
            # Layer 3: [batch_size, cube_len*4, 8, 8, 8] -> [batch_size, cube_len*2, 16, 16, 16]
            torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.cube_len * 4, self.cube_len * 2, kernel_size=4, stride=2, bias=self.args.bias, padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len * 2),
                torch.nn.ReLU()
            ),
            # Layer 4: [batch_size, cube_len*2, 16, 16, 16] -> [batch_size, cube_len, 32, 32, 32]
            torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.cube_len * 2, self.cube_len, kernel_size=4, stride=2, bias=self.args.bias, padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len),
                torch.nn.ReLU()
            ),
            # Layer 5: [batch_size, cube_len, 32, 32, 32] -> [batch_size, 1, 64, 64, 64]
            torch.nn.Sequential(
                torch.nn.ConvTranspose3d(self.cube_len, 1, kernel_size=4, stride=2, bias=self.args.bias, padding=(1, 1, 1)),
                torch.nn.Sigmoid()
            )
        ])

    def forward(self, x):
        """Forward pass through the generator"""
        # Reshape input to [batch_size, z_size, 1, 1, 1]
        out = x.view(-1, self.z_size, 1, 1, 1)

        # Pass through each layer sequentially
        for layer in self.layers:
            out = layer(out)

        return out

# Discriminator: determine if a voxel grid is real or generated
class Discriminator(ModelBase):
    def __init__(self, args):
        super(Discriminator, self).__init__(args)
        self.cube_len = args.cube_len

        # Determine padding based on cube length
        padd = (0, 0, 0)
        if self.cube_len == 32:
            padd = (1, 1, 1)

        # Create convolutional layers
        self.layers = self._build_layers(padd)

    def _build_layers(self, padd):
        """Build the discriminator layers"""
        return torch.nn.ModuleList([
            # Layer 1: [batch_size, 1, cube_len, cube_len, cube_len] -> [batch_size, cube_len, 32, 32, 32]
            torch.nn.Sequential(
                torch.nn.Conv3d(1, self.cube_len, kernel_size=4, stride=2, bias=self.args.bias, padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len),
                torch.nn.LeakyReLU(self.args.leak_value)
            ),
            # Layer 2: [batch_size, cube_len, 32, 32, 32] -> [batch_size, cube_len*2, 16, 16, 16]
            torch.nn.Sequential(
                torch.nn.Conv3d(self.cube_len, self.cube_len * 2, kernel_size=4, stride=2, bias=self.args.bias,
                                padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len * 2),
                torch.nn.LeakyReLU(self.args.leak_value)
            ),
            # Layer 3: [batch_size, cube_len*2, 16, 16, 16] -> [batch_size, cube_len*4, 8, 8, 8]
            torch.nn.Sequential(
                torch.nn.Conv3d(self.cube_len * 2, self.cube_len * 4, kernel_size=4, stride=2, bias=self.args.bias,
                                padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len * 4),
                torch.nn.LeakyReLU(self.args.leak_value)
            ),
            # Layer 4: [batch_size, cube_len*4, 8, 8, 8] -> [batch_size, cube_len*8, 4, 4, 4]
            torch.nn.Sequential(
                torch.nn.Conv3d(self.cube_len * 4, self.cube_len * 8, kernel_size=4, stride=2, bias=self.args.bias,
                                padding=(1, 1, 1)),
                torch.nn.BatchNorm3d(self.cube_len * 8),
                torch.nn.LeakyReLU(self.args.leak_value)
            ),
            # Layer 5: [batch_size, cube_len*8, 4, 4, 4] -> [batch_size, 1, 1, 1, 1]
            torch.nn.Sequential(
                torch.nn.Conv3d(self.cube_len * 8, 1, kernel_size=4, stride=2, bias=self.args.bias, padding=padd),
                torch.nn.Sigmoid()
            )
        ])

    def forward(self, x):
        """Forward pass through the discriminator"""
        # Reshape input to [batch_size, 1, cube_len, cube_len, cube_len]
        out = x.view(-1, 1, self.args.cube_len, self.args.cube_len, self.args.cube_len)

        # Pass through each layer sequentially
        for layer in self.layers:
            out = layer(out)

        return out

# Base class for all encoder variants
class EncoderBase(ModelBase):
    def __init__(self, args):
        super(EncoderBase, self).__init__(args)
        self.img_size = args.image_size
        self.batch_size = args.batch_size
        self.z_size = args.z_size

        # Create convolutional layers
        self.conv_layers = self._build_conv_layers()

        # Calculate size for fully connected layers
        input_size = self.img_size
        for i in range(5):
            input_size = self.output_features(input_size, 5, 2, 2)
        self.fc_input_size = 400 * input_size * input_size

        # Create fully connected layers for mean and log variance
        self.FC1 = torch.nn.Linear(self.fc_input_size, self.z_size)
        self.FC2 = torch.nn.Linear(self.fc_input_size, self.z_size)

    def _build_conv_layers(self):
        """Build the convolutional layers for the encoder"""
        return torch.nn.ModuleList([
            # Layer 1: [batch_size, 3, img_size, img_size] -> [batch_size, 64, img_size/2, img_size/2]
            torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, kernel_size=5, stride=2, padding=2),
                torch.nn.BatchNorm2d(64),
                torch.nn.ReLU()
            ),
            # Layer 2: [batch_size, 64, img_size/2, img_size/2] -> [batch_size, 128, img_size/4, img_size/4]
            torch.nn.Sequential(
                torch.nn.Conv2d(64, 128, kernel_size=5, stride=2, padding=2),
                torch.nn.BatchNorm2d(128),
                torch.nn.ReLU()
            ),
            # Layer 3: [batch_size, 128, img_size/4, img_size/4] -> [batch_size, 256, img_size/8, img_size/8]
            torch.nn.Sequential(
                torch.nn.Conv2d(128, 256, kernel_size=5, stride=2, padding=2),
                torch.nn.BatchNorm2d(256),
                torch.nn.ReLU()
            ),
            # Layer 4: [batch_size, 256, img_size/8, img_size/8] -> [batch_size, 512, img_size/16, img_size/16]
            torch.nn.Sequential(
                torch.nn.Conv2d(256, 512, kernel_size=5, stride=2, padding=2),
                torch.nn.BatchNorm2d(512),
                torch.nn.ReLU()
            ),
            # Layer 5: [batch_size, 512, img_size/16, img_size/16] -> [batch_size, 400, img_size/32, img_size/32]
            torch.nn.Sequential(
                torch.nn.Conv2d(512, 400, kernel_size=5, stride=2, padding=2),
                torch.nn.BatchNorm2d(400),
                torch.nn.ReLU()
            )
        ])

    def encode_single_image(self, x):
        """Encode a single image to latent space parameters"""
        # Ensure correct input shape [batch_size, 3, img_size, img_size]
        batch_size = x.size(0)
        x = x.view(batch_size, 3, self.img_size, self.img_size)

        # Pass through convolutional layers
        for layer in self.conv_layers:
            x = layer(x)

        # Flatten for fully connected layers
        x = x.view(batch_size, -1)

        # Get mean and log variance
        z_mean = self.FC1(x)
        z_log_var = self.FC2(x)

        return z_mean, z_log_var

    def reparameterize(self, mu, var):
        """Perform the reparameterization trick for VAE"""
        if self.training:
            std = torch.exp(var * 0.5)  # Convert log variance to standard deviation
            eps = var_or_cuda(std.data.new(std.size()).normal_())  # Random noise from normal distribution
            z = eps * std + mu  # Reparameterized sampling
            return z
        else:
            return mu

    def forward(self, x):
        """To be implemented by subclasses"""
        raise NotImplementedError("Subclasses must implement forward method")

# Encoder for the single-view case
class SingleViewEncoder(EncoderBase):
    def __init__(self, args):
        super(SingleViewEncoder, self).__init__(args)

    def forward(self, x):
        """Forward pass for single view encoder"""
        return self.encode_single_image(x)

# Encoder for the multi-view case
class MultiViewEncoder(EncoderBase):
    def __init__(self, args):
        super(MultiViewEncoder, self).__init__(args)
        self.combine_type = args.combine_type
        self.num_views = args.num_views

    def forward(self, images):
        """Forward pass for multi-view encoder"""
        # Initialize tensors to store results
        means = var_or_cuda(torch.zeros(self.num_views, self.batch_size, self.z_size))
        vars = var_or_cuda(torch.zeros(self.num_views, self.batch_size, self.z_size))
        zs = var_or_cuda(torch.zeros(self.num_views, self.batch_size, self.z_size))

        # Process each image
        for i, image in enumerate(images):
            image = var_or_cuda(image)
            z_mean, z_log_var = self.encode_single_image(image)
            zs[i] = self.reparameterize(z_mean, z_log_var)
            means[i] = z_mean
            vars[i] = z_log_var

        # Combine results
        return self.combine(zs), means, vars

    def combine(self, input):
        """Combine multiple view encodings based on the combination type"""
        if self.combine_type == 'mean':
            return torch.mean(input, 0)
        elif self.combine_type == 'max':
            return torch.max(input, 0)[0]
        elif self.combine_type == 'concat':
            # Concatenate views - not fully implemented
            raise NotImplementedError("Concat combination type is not implemented")
        else:
            raise ValueError(f"Unknown combination type: {self.combine_type}")
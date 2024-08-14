import torch
import argparse
from models.generator import Generator
from models.critic import Critic
from utils.dataset import CelebaDataset
from train import train_gan
from utils.visualization import show_images, morph_images


def main(args):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load dataset
    dataset = CelebaDataset(args.data_path, size=args.image_size, limit=args.dataset_limit)

    # Initialize models
    # Note: num_classes will be determined in train_gan function
    generator = Generator(args.z_dim, channels=args.channels).to(device)
    critic = Critic(channels=args.channels).to(device)

    # Train GAN
    train_gan(generator, critic, dataset, args, device)

    # Generate new faces
    noise = torch.randn(args.batch_size, args.z_dim, 1, 1, device=device)
    fake_images = generator(noise)
    show_images(fake_images, title="Generated Faces")

    # Morph images
    morph_images(generator, device, rows=4, steps=17)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Advanced GAN for Face Generation")
    parser.add_argument("--data_path", type=str, default="./data",
                        help="Path to the CelebA dataset")
    parser.add_argument("--z_dim", type=int, default=200, help="Dimension of the latent space")
    parser.add_argument("--image_size", type=int, default=128, help="Size of the images")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size for training")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--n_epochs", type=int, default=200, help="Number of training epochs")
    parser.add_argument("--dataset_limit", type=int, default=30000,
                        help="Limit on the number of images to use from the dataset")
    parser.add_argument("--channels", type=int, default=64, help="Number of base channels in the models")
    parser.add_argument("--start_size", type=int, default=4, help="Starting size for progressive growing")
    parser.add_argument("--target_size", type=int, default=128, help="Target size for progressive growing")

    args = parser.parse_args()
    main(args)
import argparse
import torch
import torchvision
from torchvision.utils import save_image
import os
from models.generator import Generator
from utils.visualization import show_images


def load_generator(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    generator = Generator(z_dim=checkpoint['z_dim'],
                          channels=checkpoint['channels'],
                          num_classes=checkpoint['num_classes'])
    generator.load_state_dict(checkpoint['model_state_dict'])
    generator.to(device)
    generator.eval()
    return generator


def generate_images(generator, num_images, z_dim, device, labels=None):
    with torch.no_grad():
        noise = torch.randn(num_images, z_dim, 1, 1, device=device)
        if labels is not None:
            labels = torch.tensor(labels, device=device)
            labels = torch.nn.functional.one_hot(labels, num_classes=generator.num_classes).float()
        generated_images = generator(noise, labels)
    return generated_images


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    print(f"Using device: {device}")

    generator = load_generator(args.model_path, device)
    print("Generator loaded successfully.")

    labels = None
    if generator.num_classes > 0:
        if args.labels:
            labels = [int(label) for label in args.labels.split(',')]
            if len(labels) != args.num_images:
                raise ValueError("Number of provided labels must match num_images")
        else:
            labels = torch.randint(0, generator.num_classes, (args.num_images,)).tolist()

    generated_images = generate_images(generator, args.num_images, generator.z_dim, device, labels)

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    # Save individual images
    for i, image in enumerate(generated_images):
        save_image(image, os.path.join(args.output_dir, f"generated_face_{i + 1}.png"), normalize=True)

    # Save grid of images
    grid = torchvision.utils.make_grid(generated_images, nrow=int(args.num_images ** 0.5), normalize=True)
    save_image(grid, os.path.join(args.output_dir, "generated_faces_grid.png"))

    print(f"Generated {args.num_images} images and saved them in {args.output_dir}")

    # Display images
    show_images(generated_images, num=min(25, args.num_images), title="Generated Faces")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate faces using a trained GAN model")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model")
    parser.add_argument("--num_images", type=int, default=16, help="Number of images to generate")
    parser.add_argument("--output_dir", type=str, default="generated_faces", help="Directory to save generated images")
    parser.add_argument("--cpu", action="store_true", help="Use CPU instead of GPU")
    parser.add_argument("--labels", type=str, help="Comma-separated list of labels for conditional generation")

    args = parser.parse_args()
    main(args)
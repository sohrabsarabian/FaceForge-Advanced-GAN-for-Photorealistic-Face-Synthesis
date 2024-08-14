import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
from utils.visualization import show_images
from collections import defaultdict


def get_gradient_penalty(real, fake, critic, device):
    alpha = torch.rand(len(real), 1, 1, 1, device=device, requires_grad=True)
    mixed_images = real * alpha + fake * (1 - alpha)
    mixed_scores = critic(mixed_images)

    gradient = torch.autograd.grad(
        inputs=mixed_images,
        outputs=mixed_scores,
        grad_outputs=torch.ones_like(mixed_scores),
        create_graph=True,
        retain_graph=True,
    )[0]

    gradient = gradient.view(len(gradient), -1)
    gradient_norm = gradient.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1) ** 2).mean()

    return gradient_penalty


def create_label_mapping(dataset):
    label_to_index = defaultdict(lambda: len(label_to_index))
    for _, label in dataset:
        label_to_index[label]
    return dict(label_to_index)


def prepare_labels(labels, label_mapping, num_classes, device):
    if labels is None:
        return None
    indices = torch.tensor([label_mapping[label] for label in labels], device=device)
    return F.one_hot(indices, num_classes=num_classes).float()


def train_gan(generator, critic, dataset, args, device):
    label_mapping = create_label_mapping(dataset)
    args.num_classes = len(label_mapping)
    print(f"Number of unique classes: {args.num_classes}")

    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    gen_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.99))
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=args.lr, betas=(0.0, 0.99))

    fixed_noise = torch.randn(64, args.z_dim, 1, 1, device=device)

    # For conditional generation
    if args.num_classes > 0:
        fixed_label_indices = torch.arange(0, args.num_classes, device=device).repeat(64 // args.num_classes + 1)[:64]
        fixed_labels = F.one_hot(fixed_label_indices, num_classes=args.num_classes).float()
    else:
        fixed_labels = None

    # Progressive growing parameters
    current_size = args.start_size

    for epoch in range(args.n_epochs):
        for real, labels in tqdm(dataloader, desc=f"Epoch {epoch + 1}/{args.n_epochs}"):
            real = real.to(device)
            current_batch_size = len(real)

            # Resize real images to current size
            real = F.interpolate(real, size=current_size)

            # Prepare labels for conditional generation
            labels = prepare_labels(labels, label_mapping, args.num_classes, device) if args.num_classes > 0 else None

            # Train Critic
            for _ in range(5):  # critic iterations
                critic_optimizer.zero_grad()
                noise = torch.randn(current_batch_size, args.z_dim, 1, 1, device=device)
                fake = generator(noise, labels)
                critic_fake_pred = critic(fake.detach(), labels)
                critic_real_pred = critic(real, labels)

                gradient_penalty = get_gradient_penalty(real, fake.detach(), critic, device)
                critic_loss = critic_fake_pred.mean() - critic_real_pred.mean() + 10 * gradient_penalty

                critic_loss.backward()
                critic_optimizer.step()

            # Train Generator
            gen_optimizer.zero_grad()
            noise = torch.randn(current_batch_size, args.z_dim, 1, 1, device=device)
            fake = generator(noise, labels)
            critic_fake_pred = critic(fake, labels)
            gen_loss = -critic_fake_pred.mean()

            gen_loss.backward()
            gen_optimizer.step()

        # Progressive growing
        if epoch % 10 == 0 and current_size < args.target_size:
            current_size *= 2
            print(f"Increasing image size to {current_size}x{current_size}")

        # Show progress
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{args.n_epochs}")
            print(f"Generator loss: {gen_loss.item():.4f}, Critic loss: {critic_loss.item():.4f}")

            with torch.no_grad():
                fake = generator(fixed_noise, fixed_labels)
                show_images(fake, title=f"Generated images after epoch {epoch + 1}")

    print("Training completed.")

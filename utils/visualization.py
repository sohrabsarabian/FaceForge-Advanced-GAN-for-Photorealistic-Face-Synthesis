import torch
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from mpl_toolkits.axes_grid1 import ImageGrid


def show_images(tensor, num=25, title=None):
    data = tensor.detach().cpu()
    grid = make_grid(data[:num], nrow=5).permute(1, 2, 0)
    plt.figure(figsize=(10, 10))
    plt.imshow(grid.clip(0, 1))
    if title:
        plt.title(title)
    plt.axis('off')
    plt.show()


def morph_images(generator, device, rows=4, steps=17):
    gen_set = []
    z_shape = [1, generator.z_dim, 1, 1]

    for _ in range(rows):
        z1, z2 = torch.randn(z_shape), torch.randn(z_shape)
        for alpha in torch.linspace(0, 1, steps):
            z = alpha * z1 + (1 - alpha) * z2
            res = generator(z.to(device))[0]
            gen_set.append(res)

    fig = plt.figure(figsize=(25, 11))
    grid = ImageGrid(fig, 111, nrows_ncols=(rows, steps), axes_pad=0.1)

    for ax, img in zip(grid, gen_set):
        ax.axis('off')
        res = img.cpu().detach().permute(1, 2, 0)
        res = (res + 1) / 2  # Denormalize
        ax.imshow(res.clip(0, 1.0))

    plt.show()

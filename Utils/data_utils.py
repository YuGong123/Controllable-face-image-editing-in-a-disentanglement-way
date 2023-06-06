import matplotlib.pyplot as plt
import torch
import os
import random
import numpy as np
from torch.utils.data import Dataset
from PIL import Image
from torchvision import transforms
from Configs import Global_Config
from torch.nn import functional as F

# to_tensor_transform = transforms.ToTensor()
to_tensor_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),
        ]
    )


def plot_single_w_image(w, generator):
    w = w.unsqueeze(0).to(Global_Config.device)
    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True
    )
    new_image = sample.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    new_image = (new_image + 1) / 2
    plt.axis('off')
    plt.imshow(new_image)
    plt.show()


def get_w_image(w, generator):
    w = w.unsqueeze(0).to(Global_Config.device)
    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True
    )
    new_image = sample.cpu().detach().numpy().transpose(0, 2, 3, 1)[0]
    new_image = (new_image + 1) / 2

    return new_image


def get_w_image2(w, generator):
    w = w.unsqueeze(0).to(Global_Config.device)
    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True
    )
    new_image = sample.cpu().detach()

    return new_image  # [-1,1],<class 'torch.Tensor'>


def get_data_by_index(idx, root_dir, postfix):
    if torch.is_tensor(idx):
        idx = idx.tolist()

    dir_idx = idx // 1000

    path = os.path.join(root_dir, str(dir_idx), str(idx) + postfix)
    if postfix == ".npy":
        data = torch.tensor(np.load(path))

    elif postfix == ".png":
        data = to_tensor_transform(Image.open(path))

    else:
        return None

    return data


class Image_W_Dataset(Dataset):
    def __init__(self, w_dir, image_dir):
        self.w_dir = w_dir
        self.image_dir = image_dir

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.w_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        w = get_data_by_index(idx, self.w_dir, ".npy")
        image = get_data_by_index(idx, self.image_dir, ".png")
        return w, image


class Image_Dataset(Dataset):
    def __init__(self, image_dir):
        self.image_dir = image_dir

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.image_dir):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        image = get_data_by_index(idx, self.image_dir, ".png")
        return image


class ID_ATTR_Dataset(Dataset):
    def __init__(self, image_dir1, image_dir2):
        self.image_dir1 = image_dir1
        self.image_dir2 = image_dir2

    def __len__(self):
        num_of_files = 0
        for base, dirs, files in os.walk(self.image_dir1):
            num_of_files += len(files)
        return num_of_files

    def __getitem__(self, idx):
        image1 = get_data_by_index(idx, self.image_dir1, ".png")
        image2 = get_data_by_index(idx, self.image_dir2, ".png")
        return image1, image2


def cycle_images_to_create_diff_order(images):
    batch_size = len(images)
    different_images = torch.empty_like(images, device=Global_Config.device)
    different_images[0] = images[batch_size - 1]
    different_images[1:] = images[:batch_size - 1]
    return different_images


def patchify_images(img_A, img_B, n_crop, min_size=1 / 8, max_size=1 / 4):  # n_crop = 8,
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size  # torch.rand(8),返回8个0-1之间的随机数， 1/8 + 1/8*a
    batch, channel, height, width = img_A.shape
    target_h = int(height * max_size)                           # target_h = 256*(1/4)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()    # crop_h = 256 * (1/8, 1/4)
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches_A = []
    patches_B = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, height - c_h)
        c_x = random.randrange(0, width - c_w)

        cropped_A = img_A[:, :, c_y: c_y + c_h, c_x: c_x + c_w]  # 将图像随机裁剪为原图的（1/8，1/4）大小
        cropped_A = F.interpolate(
            cropped_A, size=(target_h, target_w), mode="bilinear", align_corners=False  # 将图像重构成原图的1/4大小
        )

        patches_A.append(cropped_A)

        cropped_B = img_B[:, :, c_y: c_y + c_h, c_x: c_x + c_w]  # 将图像随机裁剪为原图的（1/8，1/4）大小
        cropped_B = F.interpolate(
            cropped_B, size=(target_h, target_w), mode="bilinear", align_corners=False  # 将图像重构成原图的1/4大小
        )

        patches_B.append(cropped_B)

    patches_A = torch.stack(patches_A, 1).view(-1, channel, target_h, target_w)  # 将几张图在第0维拼接
    patches_B = torch.stack(patches_B, 1).view(-1, channel, target_h, target_w)  # 将几张图在第0维拼接

    return patches_A, patches_B  # (64,3,64,64)


def patchify_image(img, n_crop, min_size=1 / 8, max_size=1 / 4):
    crop_size = torch.rand(n_crop) * (max_size - min_size) + min_size
    batch, channel, height, width = img.shape
    target_h = int(height * max_size)
    target_w = int(width * max_size)
    crop_h = (crop_size * height).type(torch.int64).tolist()
    crop_w = (crop_size * width).type(torch.int64).tolist()

    patches = []
    for c_h, c_w in zip(crop_h, crop_w):
        c_y = random.randrange(0, height - c_h)
        c_x = random.randrange(0, width - c_w)

        cropped = img[:, :, c_y : c_y + c_h, c_x : c_x + c_w]  # 将图像随机裁剪为原图的（1/8，1/4）大小
        cropped = F.interpolate(
            cropped, size=(target_h, target_w), mode="bilinear", align_corners=False  # 将图像重构成原图的1/4大小
        )

        patches.append(cropped)

    patches = torch.stack(patches, 1).view(-1, channel, target_h, target_w)  # 将几张图在第0维拼接

    return patches
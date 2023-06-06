import os
import torch
import torch.utils.data
import numpy as np
import matplotlib.pyplot as plt
from torchvision.utils import save_image
from Configs import Global_Config
from PIL import Image
from Configs.training_config import config, GENERATOR_IMAGE_SIZE
from Models.Encoders import Swap_Encoder
from torch.utils.data import DataLoader
from Models.LatentMapper import LatentMapper
from Models.StyleGan2.model import Generator
from Utils.data_utils import get_w_image, Image_W_Dataset, get_w_image2, ID_ATTR_Dataset


Swap_Encoder_PATH = "D:\PyTorchProjects\ID-disentanglement-swapping-autoencoder-Pytorch-master/experiment4/checkpoints/swap_encoder_19_225000_52.pt"
MLP_PATH = "D:\PyTorchProjects\ID-disentanglement-swapping-autoencoder-Pytorch-master/experiment4/checkpoints/mlp_19_225000_52.pt"
GENERATOR_WEIGHTS_PATH = '../pretrained_model/550000.pt'

IMAGE_DATA_DIR = r'D:\PyTorchProjects\FFT-ID-disentanglement-Pytorch-master/data/Sample_Dataset_32/small_image/'
W_DATA_DIR = r'D:\PyTorchProjects\FFT-ID-disentanglement-Pytorch-master/data/Sample_Dataset_32/small_w/'

Sample_IMAGE_DIR = '../experiment/Test_images_8x8/'
os.makedirs(Sample_IMAGE_DIR, exist_ok=True)

device = torch.device("cuda:0")

# network
# swap_encoder = Swap_Encoder.Encoder_Swap()
# mlp = LatentMapper()
generator = Generator(GENERATOR_IMAGE_SIZE, 512, 8)


# load our checkpoints
mlp = torch.load(MLP_PATH)
swap_encoder = torch.load(Swap_Encoder_PATH)

state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
generator.load_state_dict(state_dict['g_ema'], strict=False)


swap_encoder = swap_encoder.to(device)
mlp = mlp.to(device)
generator = generator.to(device)

swap_encoder = swap_encoder.eval()
mlp = mlp.eval()
generator = generator.eval()


def get_concat_vec(id_images, attr_images, swap_encoder):
    with torch.no_grad():
        id_vec, _ = swap_encoder((id_images * 2) - 1)
        _, attr_vec = swap_encoder((attr_images * 2) - 1)
        test_vec = torch.cat((torch.squeeze(id_vec), torch.squeeze(attr_vec)))
        return test_vec


test_image_dataset = ID_ATTR_Dataset(IMAGE_DATA_DIR, IMAGE_DATA_DIR)  # [0,255]的图片转换为 [0,1]的tensor
test_data_loader = DataLoader(dataset=test_image_dataset, batch_size=config['batchSize'], shuffle=True)

test_data = next(iter(test_data_loader))
test_id_images, test_attr_images = test_data    # 一次性传8张图片，可以修改batch_size的值改变图片张数


test_id_images = test_id_images.to(Global_Config.device)
test_attr_images = test_attr_images.to(Global_Config.device)


# 创建一个空白的画布
torch_figure = torch.empty((1, 3, 2304, 2304), dtype=torch.float32)  # 256*(8+1)=2304

for i in range(len(test_id_images)):
    test_id_image = test_id_images[i]  # test_id_image 为 [0,1]的tensor
    test_id_image = torch.unsqueeze(test_id_image, 0)  # (1,3,256,256)

    # 第0行显示身份图
    plot_id_image = test_id_image.cpu().detach()
    torch_figure[:, :, 0:256, (256 * (i + 1)):(256 * (i + 2))] = plot_id_image

    for j in range(len(test_attr_images)):
        test_attr_image = test_attr_images[j]  # test_attr_image 为 [0,1]的tensor
        test_attr_image = torch.unsqueeze(test_attr_image, 0)  # (1,3,256,256)

        concat_vec_mixed = get_concat_vec(test_id_image, test_attr_image, swap_encoder.eval())

        with torch.no_grad():
            mapped_concat_vec_mixed = mlp(concat_vec_mixed)
            mixed_generated_image = get_w_image2(mapped_concat_vec_mixed, generator)  # [-1,1]

        # save_image(
        #         mixed_generated_image,
        #         f'{Sample_IMAGE_DIR}/{i}{"x"}{j}.jpg',
        #         nrow=1,
        #         normalize=True,
        #         range=(-1, 1)
        #     )

        # 第(i,j)格，显示身份属性混合图
        mixed_generated_image = (mixed_generated_image + 1) / 2
        plot_mixed_generated_image = mixed_generated_image.cpu().detach()
        torch_figure[:, :, (256 * (j + 1)):(256 * (j + 2)), (256 * (i + 1)):(256 * (i + 2))] = plot_mixed_generated_image

    # 第0列显示属性图
    torch_figure[:, :, (256 * (i + 1)):(256 * (i + 2)), 0:256] = plot_id_image

# Result!
save_image(torch_figure, "%s/sample_id_attr_8x8.jpg" % (Sample_IMAGE_DIR), normalize=True, range=(0, 1))

import os
import torch
import torch.utils.data
import matplotlib.pyplot as plt
from Configs.training_config import config, GENERATOR_IMAGE_SIZE
from Models.Encoders import Swap_Encoder
from torch.utils.data import DataLoader
from Models.LatentMapper import LatentMapper
from Models.StyleGan2.model import Generator
from Utils.data_utils import get_w_image, Image_W_Dataset, cycle_images_to_create_diff_order


Swap_Encoder_PATH = "../experiment4/checkpoints/swap_encoder_19_225000_52.pt"
MLP_PATH = "../experiment4/checkpoints/mlp_19_225000_52.pt"
GENERATOR_WEIGHTS_PATH = '../pretrained_model/550000.pt'

IMAGE_DATA_DIR = r'D:\PyTorchProjects\FFT-ID-disentanglement-Pytorch-master/data/Sample_Dataset_32/small_image/'
W_DATA_DIR = r'D:\PyTorchProjects\FFT-ID-disentanglement-Pytorch-master/data/Sample_Dataset_32/small_w/'


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


# # get images

w_image_dataset = Image_W_Dataset(W_DATA_DIR, IMAGE_DATA_DIR)
data_loader = DataLoader(dataset=w_image_dataset, batch_size=config['batchSize'], shuffle=False)
data = next(iter(data_loader))
_, images = data


plot_images = images.cpu().detach().numpy().transpose(0, 2, 3, 1)


# choose idx 
id_idx = 5
attr_idx = (config['batchSize']-1) if id_idx==0 else (id_idx - 1)  # 如果id_idx = 0，attr_idx就选7，否则attr_idx就等于4

# 新图顺序：0，1，2，3，4，5，6，7
# 旧图顺序：7，0，1，2，3，4，5，6 ，start如果不指定，默认为0


# identity

plt.axis('off')
plt.imshow(plot_images[id_idx])
plt.show()


# attribute

plt.axis('off')
plt.imshow(plot_images[attr_idx])
plt.show()


# # have fun!
test_id_images = images.to(device)
test_attr_images_cycled = cycle_images_to_create_diff_order(test_id_images)
#          test_id_images顺序：0，1，2，3，4，5，6，7
# test_attr_images_cycled顺序：7，0，1，2，3，4，5，6 ，start如果不指定，默认为0


def get_concat_vec(id_images, attr_images, swap_encoder):
    with torch.no_grad():
        id_vec, _ = swap_encoder((id_images * 2) - 1)
        _, attr_vec = swap_encoder((attr_images * 2) - 1)
        test_vec = torch.cat((torch.squeeze(id_vec), torch.squeeze(attr_vec)), dim=1)
        return test_vec


concat_vec_cycled = get_concat_vec(test_id_images, test_attr_images_cycled, swap_encoder)


with torch.no_grad():
    mapped_concat_vec_cycled = mlp(concat_vec_cycled)
    cycled_generated_image = get_w_image(mapped_concat_vec_cycled[id_idx], generator)


# Result!

plt.axis('off')
plt.imshow(cycled_generated_image)
plt.show()


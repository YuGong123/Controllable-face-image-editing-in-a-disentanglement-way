from Models.StyleGan2.model import Generator
from pathlib import Path
from torchvision import utils
from tqdm import tqdm
import torch.utils.data
import numpy as np
from Configs import Global_Config

device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
GENERATOR_WEIGHTS_PATH = '../pretrained_model/550000.pt'
ID_IMAGE_DATA_DIR = '../data/eval_10000/id_small_image/'
ATTR_IMAGE_DATA_DIR = '../data/eval_10000/attr_small_image/'
W_DATA_DIR = '../data/eval_10000/small_w/'

generator = Generator(256, 512, 8).eval().to(device)
state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
generator.load_state_dict(state_dict['g_ema'], strict=False)

NUMBER_OF_IMAGES = 10000

with torch.no_grad():
    mean_latent = generator.mean_latent(4096 * 100)

counter = 0
cur_dir = 0
num_of_images_in_single_loop = 1
latents_total = []
# for i in tqdm(range(NUMBER_OF_IMAGES // num_of_images_in_single_loop)):
for i in tqdm(range(NUMBER_OF_IMAGES)):
    with torch.no_grad():
        sample_z = torch.randn(num_of_images_in_single_loop, 512, device='cuda')
        sample, latents = generator(
            [sample_z], input_is_latent=False, return_latents=True, truncation=0.5, truncation_latent=mean_latent
        )


    latents = latents.cpu().detach().numpy()


    for index in range(len(sample)):
        if (counter % 1000) == 0:
            Path(f"{W_DATA_DIR}{int(counter / 1000)}").mkdir(parents=True, exist_ok=True)
            Path(f"{ID_IMAGE_DATA_DIR}{int(counter / 1000)}").mkdir(parents=True, exist_ok=True)  # 第0个文件夹
            Path(f"{ATTR_IMAGE_DATA_DIR}{int((NUMBER_OF_IMAGES / 1000)-1-(counter / 1000))}").mkdir(parents=True, exist_ok=True)  # 第9个文件夹
            cur_dir = int(counter / 1000)
            attr_cur_dir = int((NUMBER_OF_IMAGES / 1000)-1-cur_dir)

        with open(f'{W_DATA_DIR}{cur_dir}/{counter}.npy', 'wb') as f:
            np.save(f, latents[index][0])

        utils.save_image(
            sample[index],
            f'{ID_IMAGE_DATA_DIR}{cur_dir}/{counter}.png',
            nrow=1,
            normalize=True,
            range=(-1, 1)
        )

        utils.save_image(
            sample[index],
            f'{ATTR_IMAGE_DATA_DIR}{attr_cur_dir}/{NUMBER_OF_IMAGES-1-counter}.png',  # 0,    9
            nrow=1,                                                                   # 9999, 9000
            normalize=True,
            range=(-1, 1)
        )

        counter += 1


"""
这是用来将生成训练数据集的脚本，id_images和其对应的w，attr_images是将id_images逆序在保存的结果。
"""
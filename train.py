import os
import lpips
import torch
from torchvision.utils import save_image
from Configs import Global_Config
from Configs.training_config2 import config, GENERATOR_IMAGE_SIZE
import wandb
from torch.utils.data import DataLoader
from Losses.NonAdversarialLoss import rec_loss
from Losses.freq_fourier_loss import decide_circle
from Losses.freq_pixel_loss import find_fake_freq, get_gaussian_kernel
from Models.Encoders import Swap_Encoder
from Models.Encoders.Inception import Inception
from Models.Encoders.Landmark_Encoder import Landmark_Encoder
from Models.LatentMapper import LatentMapper
from Models.StyleGan2.model import Generator
from Utils.data_utils import Image_W_Dataset, Image_Dataset, get_w_image2, patchify_images, patchify_image
import torch.utils.data
from tqdm import tqdm
from Losses import id_loss
from random import choice
from string import ascii_uppercase
from torch.nn import functional as F
from pytorch_msssim import ms_ssim

# Pre-training weight
GENERATOR_WEIGHTS_PATH = 'pretrained_model/550000.pt'  # StyleGan2 Generator for image size 256 - 550000.pt
E_ID_LOSS_PATH = 'pretrained_model/model_ir_se50.pth'
MOBILE_FACE_NET_WEIGHTS_PATH = 'pretrained_model/mobilefacenet_model_best.pth.tar'

# train and test datasets
IMAGE_DATA_DIR = r'D:\PyTorchProjects\FFT-ID-disentanglement-Pytorch-master/data/fake/small_image/'

TEST_IMAGE_DIR = r'D:\PyTorchProjects\FFT-ID-disentanglement-Pytorch-master/data/Sample_Dataset_32/small_image/'


# save models
MODELS_SAVE_DIR = 'experiment/checkpoints/'
os.makedirs(MODELS_SAVE_DIR, exist_ok=True)

# train save images
Sample_IMAGE_DIR = 'experiment/Sample_images/'
os.makedirs(Sample_IMAGE_DIR, exist_ok=True)

landmark_encoder = Landmark_Encoder.Encoder_Landmarks(MOBILE_FACE_NET_WEIGHTS_PATH)
id_loss_encoder = id_loss.IDLoss(E_ID_LOSS_PATH)
generator = Generator(GENERATOR_IMAGE_SIZE, 512, 8)
swap_encoder = Swap_Encoder.Encoder_Swap()
mlp = LatentMapper()

# load generator weights
state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
generator.load_state_dict(state_dict['g_ema'], strict=False)

landmark_encoder = landmark_encoder.to(Global_Config.device)
id_loss_encoder = id_loss_encoder.to(Global_Config.device)
generator = generator.to(Global_Config.device)
swap_encoder = swap_encoder.to(Global_Config.device)
mlp = mlp.to(Global_Config.device)

landmark_encoder = landmark_encoder.eval()
id_loss_encoder = id_loss_encoder.eval()
generator = generator.eval()
swap_encoder = swap_encoder.train()
mlp = mlp.train()


def toggle_grad(model, requires_grad):
    for p in model.parameters():
        p.requires_grad_(requires_grad)


toggle_grad(landmark_encoder, False)
toggle_grad(id_loss_encoder, False)
toggle_grad(generator, True)
toggle_grad(swap_encoder, True)
toggle_grad(mlp, True)


train_data = Image_Dataset(IMAGE_DATA_DIR)
test_dataset = Image_Dataset(TEST_IMAGE_DIR)

train_loader = DataLoader(dataset=train_data, batch_size=config['batchSize'], shuffle=False, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, drop_last=True)


optimizer_non_adv_M = torch.optim.Adam(list(mlp.parameters()) + list(swap_encoder.parameters()),
                                       lr=config['non_adverserial_lr'], betas=(config['beta1'], config['beta2']))


run_name = ''.join(choice(ascii_uppercase) for i in range(12))
run = wandb.init(project="Controllable face image editing in a disentanglement way", reinit=True, config=config, name=run_name)


def get_concat_vec(id_images, attr_images, swap_encoder):
    with torch.no_grad():
        id_vec, _ = swap_encoder((id_images * 2) - 1)
        _, attr_vec = swap_encoder((attr_images * 2) - 1)
        test_vec = torch.cat((torch.squeeze(id_vec), torch.squeeze(attr_vec)))
        return test_vec


test_data = next(iter(test_loader))
test_images = test_data    # 一次性传8张图片，可以修改batch_size的值改变图片张数

test_id_images = test_images.to(Global_Config.device)
test_attr_images = test_id_images


gauss_kernel = get_gaussian_kernel(config['gauss_size']).to(Global_Config.device)  # N 为图片张数/2，L为图片大小256
# mask_h, mask_l = decide_circle(r=config['radius'], N=int(config['batchSize'] / 2))  # batch图片张数，size图片大小
# mask_h, mask_l = mask_h.to(Global_Config.device), mask_l.to(Global_Config.device)
lpips_loss = lpips.LPIPS(net='alex').to(Global_Config.device).eval()


with tqdm(total=config['epochs'] * len(train_loader)) as pbar:
    for epoch in range(config['epochs']):
        for idx, data in enumerate(train_loader):
            Global_Config.step += 1
            images = data  # ToTensor() = [0, 1]
            real_img_freq = find_fake_freq(images.detach().clone().to(Global_Config.device), gauss_kernel)  # images的频域表示

            # 根据epoch的奇偶来划分A和B，第一轮A是前4张图，第二轮A是后4张图。这个对应的是shuffle=False
            if epoch % 2 == 0:
                real_img1, real_img2 = images.chunk(2, dim=0)  # N张输入图片分为2份, N/2为A, N/2为B.
                real_img1_freq, real_img2_freq = real_img_freq.chunk(2, dim=0)
            else:
                real_img2, real_img1 = images.chunk(2, dim=0)
                real_img2_freq, real_img1_freq = real_img_freq.chunk(2, dim=0)

            # # 根据epoch的奇偶来划分A和B，这个对应的是shuffle=True
            # real_img1, real_img2 = images.chunk(2, dim=0)  # N张输入图片分为2份, N/2为A, N/2为B.
            # real_img1_freq, real_img2_freq = real_img_freq.chunk(2, dim=0)

            real_img1 = real_img1.detach().clone().to(Global_Config.device)  # 身份图, A, [0, 1]
            real_img2 = real_img2.detach().clone().to(Global_Config.device)  # 属性图, B, [0, 1]

            real_img1_freq = real_img1_freq.detach().clone().to(Global_Config.device)
            real_img2_freq = real_img2_freq.detach().clone().to(Global_Config.device)

            # ---------------------------------
            #  Train Swap_encoder and Mlper
            # ---------------------------------

            landmark_encoder.zero_grad()
            id_loss_encoder.zero_grad()
            generator.zero_grad()
            optimizer_non_adv_M.zero_grad()

            structure1, texture1 = swap_encoder((real_img1 * 2) - 1)      # 提取A图的身份码, ID(A); 提取A图的属性码, ATTR(A)
            _, texture2 = swap_encoder((real_img2 * 2) - 1)               # 提取B图的属性码, ATTR(B)

            # structure1 = torch.squeeze(id_loss_encoder.extract_feats((real_img1 * 2) - 1))  # 提取A图的身份码, ID(A)
            # texture1 = torch.squeeze(attr_encoder(real_img1))  # 提取A图的属性码, ATTR(A)
            # texture2 = torch.squeeze(attr_encoder(real_img2))  # 提取B图的属性码, ATTR(B)

            fake_vec1 = torch.cat((structure1, texture1), dim=1)  # ID(A) + ATTR(A), 在C维度拼接, [N,C,H,W]
            fake_vec2 = torch.cat((structure1, texture2), dim=1)  # ID(A) + ATTR(B)

            fake_mlp_vec1 = mlp(fake_vec1)  # MLP[ ID(A) + ATTR(A) ]
            fake_mlp_vec2 = mlp(fake_vec2)  # MLP[ ID(A) + ATTR(B) ]

            # StyleGan_V2 生成器, out_images=[-1,1]
            fake_img1, _ = generator([fake_mlp_vec1], input_is_latent=True, return_latents=False)  # 右一图, C
            fake_img2, _ = generator([fake_mlp_vec2], input_is_latent=True, return_latents=False)  # 右二图, D

            fake_img1 = (fake_img1 + 1) / 2  # [-1, 1] --> [0, 1]
            fake_img2 = (fake_img2 + 1) / 2  # [-1, 1] --> [0, 1]

            # mse reconstruction loss
            mse_recon_loss = F.mse_loss(fake_img1, real_img1)   # fake1 = ID(A) + ATTR(A), real1 = ID(A) + ATTR(A)
            wandb.log({'mse_recon_loss_val': mse_recon_loss.detach().cpu()}, step=Global_Config.step)

            # ssim reconstruction loss, [0 to 1]
            # ssim_recon_loss = 1 - ms_ssim(fake_img1, real_img1, data_range=1, size_average=True)
            ssim_recon_loss = rec_loss(fake_img1, real_img1, config['a'])
            wandb.log({'ssim_recon_loss_val': ssim_recon_loss.detach().cpu()}, step=Global_Config.step)

            # lpips reconstruction loss  # [0, 1] --> [-1, 1]
            vgg_loss = torch.mean(lpips_loss((fake_img1 * 2) - 1, (real_img1 * 2) - 1))
            wandb.log({'vgg_loss_val': vgg_loss.detach().cpu()}, step=Global_Config.step)



            # cosine id similarity loss, [0, 1] --> [-1, 1]
            id_loss1 = id_loss_encoder((fake_img1 * 2) - 1, (real_img1 * 2) - 1)   # fake1 = ID(A) + ATTR(A), real1 = ID(A) + ATTR(A)
            wandb.log({'id_loss1_val': id_loss1.detach().cpu()}, step=Global_Config.step)

            # high frequency id similarity loss
            fake_img2_freq = find_fake_freq(fake_img2, gauss_kernel)  # 对假图做频域分解, fake2 = ID(A) + ATTR(B), real1 = ID(A) + ATTR(A)
            id_high_freq_loss2 = id_loss_encoder((fake_img2_freq[:, 3:6, :, :] * 2) - 1, (real_img1_freq[:, 3:6, :, :] * 2) - 1)
            wandb.log({'id_high_freq_loss2_val': id_high_freq_loss2.detach().cpu()}, step=Global_Config.step)


            # # landmarks_nojawline attr loss
            # _, fake_img2_landmarks_nojawline = landmark_encoder(fake_img2)
            # _, real_img2_landmarks_nojawline = landmark_encoder(real_img2)
            # landmark_loss = F.mse_loss(fake_img2_landmarks_nojawline, real_img2_landmarks_nojawline)
            # wandb.log({'landmark_loss_val': landmark_loss.detach().cpu()}, step=Global_Config.step)

            # patch attr loss
            fake_patches, real_patches = patchify_images(fake_img2, real_img2, config['n_crop'])
            attr_loss = torch.mean(lpips_loss((fake_patches * 2) - 1, (real_patches * 2) - 1))
            wandb.log({'attr_loss_val': attr_loss.detach().cpu()}, step=Global_Config.step)


            # total_loss
            total_loss = config['lambdaL2'] * mse_recon_loss + \
                         config['lambdaSSIM'] * ssim_recon_loss + \
                         config['lambdaVGG'] * vgg_loss + \
                         config['lambdaID_1'] * id_loss1 + \
                         config['lambdaID_2'] * id_high_freq_loss2 + \
                         config['lambdaATTR'] * attr_loss

            wandb.log({'total_loss_val': total_loss.detach().cpu()}, step=Global_Config.step)

            total_loss.backward()
            optimizer_non_adv_M.step()

            pbar.update(1)


            # """
            # 保存模型
            # """

            if epoch > 0 and (Global_Config.step % 5000) == 0:
                torch.save(mlp, f'{MODELS_SAVE_DIR}mlp_{epoch}_{Global_Config.step}_{int(total_loss * 100)}.pt')
                torch.save(swap_encoder, f'{MODELS_SAVE_DIR}swap_encoder_{epoch}_{Global_Config.step}_{int(total_loss*100)}.pt')


            # """
            # 测试部分
            # """

            if Global_Config.step % 200 == 0:
                # 创建一个空白的画布
                with torch.no_grad():
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

                            # 第(i,j)格，显示身份属性混合图
                            mixed_generated_image = (mixed_generated_image + 1) / 2
                            plot_mixed_generated_image = mixed_generated_image.cpu().detach()
                            torch_figure[:, :, (256 * (j + 1)):(256 * (j + 2)), (256 * (i + 1)):(256 * (i + 2))] = plot_mixed_generated_image

                        # 第0列显示属性图
                        torch_figure[:, :, (256 * (i + 1)):(256 * (i + 2)), 0:256] = plot_id_image

                    # Result!
                    save_image(torch_figure, "%s/sample_%s_%s.jpg" % (Sample_IMAGE_DIR, epoch, Global_Config.step),
                               normalize=True, range=(0, 1))

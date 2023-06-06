# Controllable face image editing in a disentanglement way - Implement in Pytorch with StyleGAN2



## Description

Pytorch implementation of the paper *Controllable face image editing in a disentanglement way* for both training and evaluation, with StyleGAN 2.

- Reference Code: https://github.com/danielroich/ID-disentanglement-Pytorch
- Reference papers: https://arxiv.org/abs/2005.07728

Our code is more complicated, if you want to run it on your own computer, you need to re-modify the dataset path. If you have other things that you don't understand, you can also refer to the above **Reference Code**.



## Setup

We used several **pretrained models**: 
- StyleGan2 Generator for image size 256 - 550000.pt
- ID Encoder - model_ir_se50.pth
- Landmarks Detection - mobilefacenet_model_best.pth.tar

Weight files attached at this [Drive folder](https://drive.google.com/drive/folders/18K5YBBJRiCIradtttlLcdtSyLUo3cUI5?usp=sharing).

You can also find at the above link our **environment.yml** file to create a relevant conda environment.



## Datasets  

The dataset is comprised of StyleGAN 2 generated images. 

We randomly sample 70,000 Gaussian noises $z$ and then map them to $\mathcal{W}$, the latent space of the pre-trained generator G, which in turn generates the resulting images $G(w)$ . Actually, to improve the average quality of the generated images, we first calculate the average vector $\bar{w} $ from the latent space $\mathcal{W}$ of the selected pretrained StyleGAN2 generator. Finally, the resulting images $G(w + \bar{w} )$  are used as our training dataset. The test dataset is generated in the same way.

You can use Utils/**data_creator.py**  to generate the dataset in the paper.



## **Architecture** 

![Architecture](./Architecture.jpg)

Green blocks represent network components, solid lines represent data flows, and dotted lines represent losses. The identity image $x^{1}$ and attribute image $x^{2}$ are sent to the same encoder. Identity branch  $E_{id}$  is used to extract identity code  $z_{id}$, and attribute branch $E_{attr}$ is used to extract attribute code $z_{attr}$. The new combination code $z = ({z_{id}},{z_{attr}})$ is mapped through a mapper network M to $\mathcal{W}$, the latent space of the pre-trained generator G, which in turn generates a reconstructed image (above) or a mixed image (below). $L_{mix}$ and $L_{lpips}$ are reconstruction losses used to preserve pixel-level information and learn perceptual similarity, respectively. $L_{id_2}$  is an identity preservation loss that uses the identity identifier R to learn identity feature by calculating cosine similarity of two pictures $\hat{x}^{2}$ and $x^{1}$. $L_{attr}$  is an attribute preservation loss that uses the image clipper C to achieve partial similarity between  $\hat{x}^{2}$ and $x^{2}$.



## Training

Note, I recommend that you use Utils/**data_creator.py** to generate training dataset, which saves you the trouble of paths.

To train the model run **train.py**, you can change parameters in **Configs/** folder.



## Checkpoints

Our pretrained checkpoint (swap_encoder.pt and mlp.pt) attached at this https://pan.baidu.com/s/1uI49sT58jadTV8ZPbPAVIQ 
Extraction code：9hq6 . Or you can get your own checkpoints by  **train.py**.



## Inference

Try **Inference.py** notebook to disentangle identity from attributes by yourself.



## Results

![Results](./Results.jpg)



## Web-Demo

![Web_Demo](./Web_Demo/Web_Demo.jpg)
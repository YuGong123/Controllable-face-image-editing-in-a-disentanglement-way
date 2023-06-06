# coding:utf-8


import torch
from flask_cors import CORS
from torchvision import transforms
from torchvision.utils import save_image
from Models.StyleGan2.model import Generator
from PIL import Image
from flask import Flask, render_template, request, redirect, url_for, make_response, jsonify, send_from_directory
from werkzeug.utils import secure_filename
import os
from datetime import timedelta

app = Flask(__name__)
CORS(app)
# 设置静态文件缓存过期时间
app.send_file_max_age_default = timedelta(seconds=1)


"""ID-disentanglement-swapping-autoencoder"""
device = torch.device("cuda:0")

# checkpoints
Swap_Encoder_PATH = "./checkpoints/swap_encoder_19_225000_52.pt"
MLP_PATH = "./checkpoints/mlp_19_225000_52.pt"
GENERATOR_WEIGHTS_PATH = './checkpoints/550000.pt'

# network
# swap_encoder = Swap_Encoder.Encoder_Swap()
# mlp = LatentMapper()
generator = Generator(256, 512, 8)

# load our checkpoints
mlp = torch.load(MLP_PATH)
swap_encoder = torch.load(Swap_Encoder_PATH)
state_dict = torch.load(GENERATOR_WEIGHTS_PATH)
generator.load_state_dict(state_dict['g_ema'], strict=False)

# cuda:0
swap_encoder = swap_encoder.to(device)
mlp = mlp.to(device)
generator = generator.to(device)

# model.eval()
swap_encoder = swap_encoder.eval()
mlp = mlp.eval()
generator = generator.eval()


# to_tensor_transform = transforms.ToTensor()
to_tensor_transform = transforms.Compose(
        [
            transforms.Resize((256, 256), Image.BICUBIC),
            transforms.ToTensor(),  # 取值范围为[0, 255]的PIL.Image，转换成形状为[C, H, W]，取值范围是[0, 1.0]的torch.FloadTensor
        ]
    )


def get_concat_vec(id_images, attr_images, swap_encoder):
    with torch.no_grad():
        id_vec, _ = swap_encoder((id_images * 2) - 1)
        _, attr_vec = swap_encoder((attr_images * 2) - 1)
        test_vec = torch.cat((torch.squeeze(id_vec), torch.squeeze(attr_vec)))
        return test_vec


def get_w_image2(w, generator):
    w = w.unsqueeze(0).to(device)
    sample, latents = generator(
        [w], input_is_latent=True, return_latents=True
    )
    new_image = sample.cpu().detach()

    return new_image  # [-1,1],<class 'torch.Tensor'>
"""ID-disentanglement-swapping-autoencoder"""



# 设置允许的文件格式
ALLOWED_EXTENSIONS = set(['png', 'jpg', 'JPG', 'PNG', 'bmp'])


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1] in ALLOWED_EXTENSIONS


def return_img_stream(img_local_path):
    """
    工具函数:
    获取本地图片流
    :param img_local_path:文件单张图片的本地绝对路径
    :return: 图片流
    """
    import base64
    img_stream = ''
    with open(img_local_path, 'rb') as img_f:
        img_stream = img_f.read()
        img_stream = base64.b64encode(img_stream).decode()
    return img_stream



@app.route('/', methods=['POST', 'GET'])  # 添加路由
def upload():
    if request.method == 'POST':
        f1 = request.files['file1']
        f2 = request.files['file2']

        if not (f1 and allowed_file(f1.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        if not (f2 and allowed_file(f2.filename)):
            return jsonify({"error": 1001, "msg": "请检查上传的图片类型，仅限于png、PNG、jpg、JPG、bmp"})

        basepath = os.path.dirname(__file__)  # 当前文件所在路径

        upload_path1 = os.path.join(basepath, 'static/images', secure_filename(f1.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径
        upload_path2 = os.path.join(basepath, 'static/images', secure_filename(f2.filename))  # 注意：没有的文件夹一定要先创建，不然会提示没有该路径

        f1.save(upload_path1)
        f2.save(upload_path2)


        ##############################################################################################
        id_image = to_tensor_transform(Image.open(upload_path1))
        attr_image = to_tensor_transform(Image.open(upload_path2))
        id_image = id_image.unsqueeze(0).detach().clone().to(device)
        attr_image = attr_image.unsqueeze(0).detach().clone().to(device)
        # print(id_image.shape, type(id_image))

        concat_vec = get_concat_vec(id_image, attr_image, swap_encoder)

        with torch.no_grad():
            mapped_concat_vec = mlp(concat_vec)
            generated_image = get_w_image2(mapped_concat_vec, generator)  # [-1,1],<class 'torch.Tensor'>
            # print(generated_image.shape, type(generated_image), torch.max(generated_image), torch.min(generated_image))


        download_path = os.path.join(basepath, 'static/images', 'test.jpg')

        save_image(
            generated_image,
            download_path,
            nrow=1,
            normalize=True,
            range=(-1, 1)
        )
        ##############################################################################################

        id_img_stream = return_img_stream(upload_path1)
        attr_img_stream = return_img_stream(upload_path2)
        mixed_img_stream = return_img_stream(download_path)

        return render_template('predict.html',
                               id_img_stream=id_img_stream,
                               attr_img_stream=attr_img_stream,
                               mixed_img_stream=mixed_img_stream
                               )

    return render_template('index.html')


# 注意，下载路径 "download_path" 是写死的
@app.route('/download/', methods=['GET'])
def download():
    return send_from_directory('static/images', 'test.jpg', as_attachment=True)


if __name__ == '__main__':
    # app.debug = True
    app.run(host='0.0.0.0', port=5000, debug=True)

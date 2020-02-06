import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import glob
import matplotlib.pyplot as plt

# 已训练好的styleGAN模型路径
Model = config.Model

synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True), minibatch_size=8)
_Gs_cache = dict()


# 已训练好的styleGAN模型路径
def load_Gs(model):
    if model not in _Gs_cache:
        model_file = glob.glob(Model)
        if len(model_file) == 1:
            model_file = open(model_file[0], "rb")
        else:
            raise Exception('Failed to find the model')

        _G, _D, Gs = pickle.load(model_file)
        # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
        # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
        # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

        # Print network details.
        Gs.print_layers()

        _Gs_cache[model] = Gs
    return _Gs_cache[model]


# Changing Style
# 用目标图像的src_dlatents的一部分替换原图像的dst_dlatents的对应部分，
# 然后用Gs.components.synthesis.run（）函数生成风格混合后的图像
def change_style_figure(save_name, mix1, mix2, Gs, style_ranges):
    os.makedirs(config.generated_dir, exist_ok=True)
    save_path = os.path.join(config.generated_dir, save_name + '.png')
    print(save_path)

    os.makedirs(config.dlatents_dir, exist_ok=True)
    src = np.load(os.path.join(config.dlatents_dir, mix1 + '.npy'))
    dst = np.load(os.path.join(config.dlatents_dir, mix2 + '.npy'))

    src_dlatents = np.expand_dims(src, axis=0)
    dst_dlatents = np.expand_dims(dst, axis=0)

    # 从dlatents生成图像
    src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=config.randomize_noise, **synthesis_kwargs)
    dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=config.randomize_noise, **synthesis_kwargs)

    # 画空白图
    Style_No = len(style_ranges)
    w = src_images.shape[1]
    h = src_images.shape[2]
    canvas = PIL.Image.new('RGB', (w * (Style_No + 2), h), 'white')

    # 在画布的第一格画原图像
    canvas.paste(PIL.Image.fromarray(src_images[0], 'RGB'), (0, 0))

    # 在画布逐行绘制图像
    # 最后一格绘制目标图像
    canvas.paste(PIL.Image.fromarray(dst_images[0], 'RGB'), ((Style_No + 1) * w, 0))

    # 将源图像复制N份，构成新数组
    row_dlatents = np.stack([src_dlatents[0]] * Style_No)

    # 用dst_dlatents的指定列替换row_dlatents的指定列，数据混合
    for i in range(Style_No):
        row_dlatents[i, style_ranges[i]] = dst_dlatents[0, style_ranges[i]]

    # 调用用Gs.components.synthesis.run（）函数生成风格混合后的图像
    row_images = Gs.components.synthesis.run(row_dlatents, randomize_noise=config.randomize_noise, **synthesis_kwargs)

    # 在画布上逐列绘制风格混合后的图像
    for col, image in enumerate(list(row_images)):
        canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * w, 0))

    canvas.show()
    canvas.save(save_path)


# 将图像的潜码与控制向量混合以定向修改图像
def move_and_show(latent_vector, direction, coeffs, Gs):
    fig, ax = plt.subplots(1, len(coeffs), figsize=(15, 10), dpi=80)
    for i, coeff in enumerate(coeffs):
        new_latent_vector = latent_vector.copy()
        new_latent_vector[:8] = (latent_vector + coeff * direction)[:8]
        img_array = Gs.components.synthesis.run(np.expand_dims(new_latent_vector, axis=0), randomize_noise=config.randomize_noise, **synthesis_kwargs)
        img = PIL.Image.fromarray(img_array[0], 'RGB')
        ax[i].imshow(img)
        ax[i].set_title('Coeff: %0.1f' % coeff)
    [x.axis('off') for x in ax]
    plt.show()


# 测试生成器
def sample(num):
    latents = np.random.randn(num**2, Gs.input_shape[1])
    src_images = Gs.run(latents, None, **synthesis_kwargs)

    w = Gs.output_shape[2]
    h = Gs.output_shape[3]
    canvas = PIL.Image.new('RGB', (w*num, h*num), 'white')
    # 在画布的第一格画原图像
    for i in range(num):
        for j in range(num):
            canvas.paste(PIL.Image.fromarray(src_images[i*num+j], 'RGB'), (i*w, j*h))
    canvas.show()


def reshape_img(img_path, image_size):
    img = PIL.Image.open(img_path).convert('RGB').resize((image_size, image_size), PIL.Image.LANCZOS)
    img.save(img_path)

tflib.init_tf()
Gs = load_Gs(Model)

age_direction = np.load('ffhq_dataset/latent_directions/age.npy')
angle_direction = np.load('ffhq_dataset/latent_directions/angle_horizontal.npy')
gender_direction = np.load('ffhq_dataset/latent_directions/gender.npy')
eyes_direction = np.load('ffhq_dataset/latent_directions/eyes_open.npy')
glasses_direction = np.load('ffhq_dataset/latent_directions/glasses.npy')
smile_direction = np.load('ffhq_dataset/latent_directions/smile.npy')


change_style_figure('change-style-figure.png', 'c05', 'c15', Gs,
                    style_ranges=[range(0, 4), range(4, 8), range(8, 16)])  # 混合
                    # style_ranges=[range(i, 16) for i in range(9, 0, -3)])  # 逐渐
                    # style_ranges=[range(3, 4), range(4, 5), range(5, 6), range(6, 7), range(7, 8), range(8, 9), range(9, 10)])

latent = np.load(os.path.join(config.dlatents_dir, '17_04.npy'))
move_and_show(latent, age_direction, range(-40, 45, 10), Gs)  # age
move_and_show(latent, gender_direction, range(-60, 75, 15), Gs)  # gender
move_and_show(latent, smile_direction, range(-15, 20, 5), Gs)  # smile


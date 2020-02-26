import os
import argparse
import pickle
import time

from tqdm import tqdm
import PIL.Image
import numpy as np

import config
import dnnlib
import dnnlib.tflib as tflib
from encoder_s1.generator_model import Generator
from encoder_s1.perceptual_model import PerceptualModel, load_images
from keras.models import load_model

import glob
import random


def split_to_batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


def main():
    parser = argparse.ArgumentParser(
        description='Find latent representation of reference images using perceptual losses',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('src_dir', help='Directory with images for encoding')
    parser.add_argument('generated_images_dir', help='Directory for storing generated images')
    parser.add_argument('dlatent_dir', help='Directory for storing dlatent representations')
    parser.add_argument('--data_dir', default='data', help='Directory for storing optional models')
    parser.add_argument('--mask_dir', default='masks', help='Directory for storing optional masks')
    parser.add_argument('--load_last', default='', help='Start with embeddings from directory')
    parser.add_argument('--dlatent_avg', default='',
                        help='Use dlatent from file specified here for truncation instead of dlatent_avg from Gs')
    parser.add_argument('--model_url', default=config.Model,
                        help='Fetch a StyleGAN model to train on from this URL')  # karras2019stylegan-ffhq-1024x1024.pkl
    parser.add_argument('--model_res', default=1024, help='The dimension of images in the StyleGAN model', type=int)
    parser.add_argument('--batch_size', default=1, help='Batch size for generator and perceptual model', type=int)

    # Perceptual model params
    parser.add_argument('--image_size', default=256, help='Size of images for perceptual model', type=int)
    parser.add_argument('--resnet_image_size', default=224, help='Size of images for the Resnet model', type=int)
    parser.add_argument('--lr', default=0.02, help='Learning rate for perceptual model', type=float)
    parser.add_argument('--decay_rate', default=0.9, help='Decay rate for learning rate', type=float)
    parser.add_argument('--iterations', default=100, help='Number of optimization steps for each batch', type=int)
    parser.add_argument('--decay_steps', default=10,
                        help='Decay steps for learning rate decay (as a percent of iterations)', type=float)
    parser.add_argument('--load_effnet', default='data/finetuned_effnet.h5',
                        help='Model to load for EfficientNet approximation of dlatents')
    parser.add_argument('--load_resnet', default='data/resnet_18.h5',
                        help='Model to load for ResNet approximation of dlatents')

    # Loss function options
    parser.add_argument('--use_vgg_loss', default=0.4, help='Use VGG perceptual loss; 0 to disable, > 0 to scale.',
                        type=float)
    parser.add_argument('--use_vgg_layer', default=9, help='Pick which VGG layer to use.', type=int)
    parser.add_argument('--use_pixel_loss', default=1.5,
                        help='Use logcosh image pixel loss; 0 to disable, > 0 to scale.', type=float)
    parser.add_argument('--use_mssim_loss', default=100, help='Use MS-SIM perceptual loss; 0 to disable, > 0 to scale.',
                        type=float)
    parser.add_argument('--use_lpips_loss', default=100, help='Use LPIPS perceptual loss; 0 to disable, > 0 to scale.',
                        type=float)
    parser.add_argument('--use_l1_penalty', default=1, help='Use L1 penalty on latents; 0 to disable, > 0 to scale.',
                        type=float)

    # Generator params
    parser.add_argument('--randomize_noise', default=False, help='Add noise to dlatents during optimization', type=bool)
    parser.add_argument('--tile_dlatents', default=False, help='Tile dlatents to use a single vector at each scale',
                        type=bool)
    parser.add_argument('--clipping_threshold', default=2.0,
                        help='Stochastic clipping of gradient values outside of this threshold', type=float)

    # Masking params
    parser.add_argument('--load_mask', default=False, help='Load segmentation masks', type=bool)
    parser.add_argument('--face_mask', default=False, help='Generate a mask for predicting only the face area',
                        type=bool)
    parser.add_argument('--use_grabcut', default=True,
                        help='Use grabcut algorithm on the face mask to better segment the foreground', type=bool)
    parser.add_argument('--scale_mask', default=1.5, help='Look over a wider section of foreground for grabcut',
                        type=float)

    # Video params
    parser.add_argument('--video_dir', default='videos', help='Directory for storing training videos')
    parser.add_argument('--output_video', default=False, help='Generate videos of the optimization process', type=bool)
    parser.add_argument('--video_codec', default='MJPG', help='FOURCC-supported video codec name')
    parser.add_argument('--video_frame_rate', default=24, help='Video frames per second', type=int)
    parser.add_argument('--video_size', default=512, help='Video size in pixels', type=int)
    parser.add_argument('--video_skip', default=1, help='Only write every n frames (1 = write every frame)', type=int)

    # 获取到基本设置时，如果运行命令中传入了之后才会获取到的其他配置，不会报错；而是将多出来的部分保存起来，留到后面使用
    args, other_args = parser.parse_known_args()

    # learning rate衰减的steps
    args.decay_steps *= 0.01 * args.iterations  # Calculate steps as a percent of total iterations

    if args.output_video:
        import cv2
        synthesis_kwargs = dict(output_transform=dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=False),
                                minibatch_size=args.batch_size)

    # 找到src_dir下所有图片文件，加入ref_images列表（即：源图的列表；只有一个图片也可以）
    ref_images = [os.path.join(args.src_dir, x) for x in os.listdir(args.src_dir)]
    ref_images = list(filter(os.path.isfile, ref_images))

    if len(ref_images) == 0:
        raise Exception('%s is empty' % args.src_dir)

    # 创建工作目录
    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.mask_dir, exist_ok=True)
    os.makedirs(args.generated_images_dir, exist_ok=True)
    os.makedirs(args.dlatent_dir, exist_ok=True)
    os.makedirs(args.video_dir, exist_ok=True)

    # Initialize generator and perceptual model
    tflib.init_tf()
    # 加载StyleGAN模型
    model_file = glob.glob(args.model_url)
    if len(model_file) == 1:
        model_file = open(model_file[0], "rb")
    else:
        raise Exception('Failed to find the model')
    generator_network, discriminator_network, Gs_network = pickle.load(model_file)

    # 加载Generator类，参与构建VGG16 perceptual model，用于调用（说是生成，更好理解）generated_image
    # generated_image通过perceptual_model转化为generated_img_features，参与计算loss
    generator = Generator(Gs_network, args.batch_size, clipping_threshold=args.clipping_threshold,
                          tiled_dlatent=args.tile_dlatents, model_res=args.model_res,
                          randomize_noise=args.randomize_noise)
    if (args.dlatent_avg != ''):
        generator.set_dlatent_avg(np.load(args.dlatent_avg))

    perc_model = None
    if (args.use_lpips_loss > 0.00000001):  # '--use_lpips_loss', default = 100
        # 加载VGG16 perceptual模型
        model_file = glob.glob('./models/vgg16_zhang_perceptual.pkl')
        if len(model_file) == 1:
            model_file = open(model_file[0], "rb")
        else:
            raise Exception('Failed to find the model')
        perc_model = pickle.load(model_file)

    # 创建VGG16 perceptual模型
    perceptual_model = PerceptualModel(args, perc_model=perc_model, batch_size=args.batch_size)
    perceptual_model.build_perceptual_model(generator)

    ff_model = None
    # Optimize (only) dlatents by minimizing perceptual loss between reference and generated images in feature space
    # tqdm 是一个快速，可扩展的Python进度条，可以在 Python 长循环中添加一个进度提示信息
    # 把ref_images分割为若干批次，每个批次的大小为args.batch_size，分批使用perceptual_model.optimize()求解每个源图的dlatents的最优解
    # 对每一个源图，优化迭代的过程是从一个初始dlatents开始，在某个空间内，按正态分布取值，使用Adam优化器，逐步寻找使loss最小的dlatents，即：stochastic clipping方法
    for images_batch in tqdm(split_to_batches(ref_images, args.batch_size), total=len(ref_images) // args.batch_size):
        print('开始计时')
        starttime = time.time()
        # 读取每个批次中的文件名
        names = [os.path.splitext(os.path.basename(x))[0] for x in images_batch]
        if args.output_video:
            video_out = {}
            for name in names:
                video_out[name] = cv2.VideoWriter(os.path.join(args.video_dir, f'{name}.avi'),
                                                  cv2.VideoWriter_fourcc(*args.video_codec), args.video_frame_rate,
                                                  (args.video_size, args.video_size))

        # 给源图及源图用VGG16生成的features赋值(这是计算loss的基准)
        perceptual_model.set_reference_images(images_batch)
        dlatents = None
        if (args.load_last != ''):  # load previous dlatents for initialization
            for name in names:
                dl = np.expand_dims(np.load(os.path.join(args.load_last, f'{name}.npy')), axis=0)
                if (dlatents is None):
                    dlatents = dl
                else:
                    dlatents = np.vstack((dlatents, dl))
        else:
            if (ff_model is None):
                if os.path.exists(args.load_resnet):
                    print("Loading ResNet Model:")
                    ff_model = load_model(args.load_resnet)
                    from keras.applications.resnet50 import preprocess_input
            if (ff_model is None):
                if os.path.exists(args.load_effnet):
                    import efficientnet
                    print("Loading EfficientNet Model:")
                    ff_model = load_model(args.load_effnet)
                    from efficientnet import preprocess_input
            if (ff_model is not None):  # predict initial dlatents with ResNet model
                dlatents = ff_model.predict(
                    preprocess_input(load_images(images_batch, image_size=args.resnet_image_size)))
        # 设置用于perceptual_model优化迭代的初始值dlatents，它是用resnet50或者efficientnet从源图预测得到的
        if dlatents is not None:
            generator.set_dlatents(dlatents)
        # 对每一个源图，用tqdm构造进度条，显示优化迭代的过程
        op = perceptual_model.optimize(generator.dlatent_variable, iterations=args.iterations)
        pbar = tqdm(op, leave=False, total=args.iterations)
        vid_count = 0
        best_loss = None
        best_dlatent = None
        # 用stochastic clipping方法，使用VGG16 perceptual_model进行优化迭代，迭代次数为iterations=args.iterations
        endtime = time.time()
        print('开始迭代时间为:', round(endtime - starttime, 2), 'secs')
        for loss_dict in pbar:
            pbar.set_description(" ".join(names) + ": " + "; ".join(["{} {:.4f}".format(k, v)
                                                                     for k, v in loss_dict.items()]))
            if best_loss is None or loss_dict["loss"] < best_loss:
                best_loss = loss_dict["loss"]
                best_dlatent = generator.get_dlatents()
            if args.output_video and (vid_count % args.video_skip == 0):
                batch_frames = generator.generate_images()
                for i, name in enumerate(names):
                    video_frame = PIL.Image.fromarray(batch_frames[i], 'RGB').resize((args.video_size, args.video_size),
                                                                                     PIL.Image.LANCZOS)
                    video_out[name].write(cv2.cvtColor(np.array(video_frame).astype('uint8'), cv2.COLOR_RGB2BGR))
            # 用stochastic clip方法更新dlatent_variable
            generator.stochastic_clip_dlatents()
        print(" ".join(names), " Loss {:.4f}".format(best_loss))

        if args.output_video:
            for name in names:
                video_out[name].release()

        # Generate images from found dlatents and save them
        generator.set_dlatents(best_dlatent)
        generated_images = generator.generate_images()
        generated_dlatents = generator.get_dlatents()
        endtime = time.time()
        print('计算完成时间为:', round(endtime - starttime, 2), 'secs')
        for img_array, dlatent, img_name in zip(generated_images, generated_dlatents, names):
            img = PIL.Image.fromarray(img_array, 'RGB')
            img.save(os.path.join(args.generated_images_dir, f'{img_name}.png'), 'PNG')
            np.save(os.path.join(args.dlatent_dir, f'{img_name}.npy'), dlatent)

        generator.reset_dlatents()
        endtime = time.time()
        print('写入完成时间为:', round(endtime - starttime, 2), 'secs')


if __name__ == "__main__":
    main()


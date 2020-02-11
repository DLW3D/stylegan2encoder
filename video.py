import os
import pickle
import numpy as np
import PIL.Image
import dnnlib
import dnnlib.tflib as tflib
import config
import scipy
import dnnlib.tflib as tflib
import math
import moviepy.editor
from numpy import linalg
import numpy as np
import pickle


def main():
    tflib.init_tf()

    # Load pre-trained network.
    # url = 'https://drive.google.com/uc?id=1MEGjdvVpUsu1jB4zrXZN7Y4kBBOzizDQ'
    # with dnnlib.util.open_url(url, cache_dir=config.cache_dir) as f:
    ## NOTE: insert model here:
    _G, _D, Gs = pickle.load(open(config.Model, "rb"))
    # _G = Instantaneous snapshot of the generator. Mainly useful for resuming a previous training run.
    # _D = Instantaneous snapshot of the discriminator. Mainly useful for resuming a previous training run.
    # Gs = Long-term average of the generator. Yields higher-quality results than the instantaneous snapshot.

    grid_size = [2, 2]
    image_shrink = 1
    image_zoom = 1
    duration_sec = 60.0
    smoothing_sec = 1.0
    mp4_fps = 20
    mp4_codec = 'libx264'
    mp4_bitrate = '5M'
    random_seed = 404
    mp4_file = 'results/random_grid_%s.mp4' % random_seed
    minibatch_size = 8

    num_frames = int(np.rint(duration_sec * mp4_fps))
    random_state = np.random.RandomState(random_seed)

    # Generate latent vectors
    shape = [num_frames, np.prod(grid_size)] + Gs.input_shape[1:]  # [frame, image, channel, component]
    all_latents = random_state.randn(*shape).astype(np.float32)
    import scipy
    all_latents = scipy.ndimage.gaussian_filter(all_latents,
                                                [smoothing_sec * mp4_fps] + [0] * len(Gs.input_shape), mode='wrap')
    all_latents /= np.sqrt(np.mean(np.square(all_latents)))

    def create_image_grid(images, grid_size=None):
        assert images.ndim == 3 or images.ndim == 4
        num, img_h, img_w, channels = images.shape

        if grid_size is not None:
            grid_w, grid_h = tuple(grid_size)
        else:
            grid_w = max(int(np.ceil(np.sqrt(num))), 1)
            grid_h = max((num - 1) // grid_w + 1, 1)

        grid = np.zeros([grid_h * img_h, grid_w * img_w, channels], dtype=images.dtype)
        for idx in range(num):
            x = (idx % grid_w) * img_w
            y = (idx // grid_w) * img_h
            grid[y: y + img_h, x: x + img_w] = images[idx]
        return grid

    # Frame generation func for moviepy.
    def make_frame(t):
        frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
        latents = all_latents[frame_idx]
        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
        images = Gs.run(latents, None, truncation_psi=0.7,
                        randomize_noise=False, output_transform=fmt)

        grid = create_image_grid(images, grid_size)
        if image_zoom > 1:
            grid = scipy.ndimage.zoom(grid, [image_zoom, image_zoom, 1], order=0)
        if grid.shape[2] == 1:
            grid = grid.repeat(3, 2)  # grayscale => RGB
        return grid

    # Generate video.
    import moviepy.editor
    video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    video_clip.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)

    ####################################################################################################################
    # # import scipy
    # # coarse
    # duration_sec = 60.0
    # smoothing_sec = 1.0
    # mp4_fps = 20
    #
    # num_frames = int(np.rint(duration_sec * mp4_fps))
    # random_seed = 500
    # random_state = np.random.RandomState(random_seed)
    #
    # w = 512
    # h = 512
    # # src_seeds = [601]
    # dst_seeds = [700]
    # style_ranges = ([0] * 7 + [range(8, 16)]) * len(dst_seeds)
    #
    # fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    # synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)
    #
    # shape = [num_frames] + Gs.input_shape[1:]  # [frame, image, channel, component]
    # src_latents = random_state.randn(*shape).astype(np.float32)
    # src_latents = scipy.ndimage.gaussian_filter(src_latents,
    #                                             smoothing_sec * mp4_fps,
    #                                             mode='wrap')
    # src_latents /= np.sqrt(np.mean(np.square(src_latents)))
    #
    # dst_latents = np.stack(np.random.RandomState(seed).randn(Gs.input_shape[1]) for seed in dst_seeds)
    #
    # src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
    # dst_dlatents = Gs.components.mapping.run(dst_latents, None)  # [seed, layer, component]
    # src_images = Gs.components.synthesis.run(src_dlatents, randomize_noise=False, **synthesis_kwargs)
    # dst_images = Gs.components.synthesis.run(dst_dlatents, randomize_noise=False, **synthesis_kwargs)
    #
    # canvas = PIL.Image.new('RGB', (w * (len(dst_seeds) + 1), h * 2), 'white')
    #
    # for col, dst_image in enumerate(list(dst_images)):
    #     canvas.paste(PIL.Image.fromarray(dst_image, 'RGB'), ((col + 1) * h, 0))
    #
    # def make_frame(t):
    #     frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
    #     src_image = src_images[frame_idx]
    #     canvas.paste(PIL.Image.fromarray(src_image, 'RGB'), (0, h))
    #
    #     for col, dst_image in enumerate(list(dst_images)):
    #         col_dlatents = np.stack([dst_dlatents[col]])
    #         col_dlatents[:, style_ranges[col]] = src_dlatents[frame_idx, style_ranges[col]]
    #         col_images = Gs.components.synthesis.run(col_dlatents, randomize_noise=False, **synthesis_kwargs)
    #         for row, image in enumerate(list(col_images)):
    #             canvas.paste(PIL.Image.fromarray(image, 'RGB'), ((col + 1) * h, (row + 1) * w))
    #     return np.array(canvas)
    #
    # # Generate video.
    # import moviepy.editor
    # mp4_file = 'results/interpolate.mp4'
    # mp4_codec = 'libx264'
    # mp4_bitrate = '5M'
    #
    # video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    # video_clip.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)
    #
    ####################################################################################################################
    # import scipy
    #
    # duration_sec = 60.0
    # smoothing_sec = 1.0
    # mp4_fps = 20
    #
    # num_frames = int(np.rint(duration_sec * mp4_fps))
    # random_seed = 503
    # random_state = np.random.RandomState(random_seed)
    #
    # w = 512
    # h = 512
    # style_ranges = [range(6, 16)]
    #
    # fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)
    # synthesis_kwargs = dict(output_transform=fmt, truncation_psi=0.7, minibatch_size=8)
    #
    # shape = [num_frames] + Gs.input_shape[1:]  # [frame, image, channel, component]
    # src_latents = random_state.randn(*shape).astype(np.float32)
    # src_latents = scipy.ndimage.gaussian_filter(src_latents,
    #                                             smoothing_sec * mp4_fps,
    #                                             mode='wrap')
    # src_latents /= np.sqrt(np.mean(np.square(src_latents)))
    #
    # dst_latents = np.stack([random_state.randn(Gs.input_shape[1])])
    #
    # src_dlatents = Gs.components.mapping.run(src_latents, None)  # [seed, layer, component]
    # dst_dlatents = Gs.components.mapping.run(dst_latents, None)  # [seed, layer, component]
    #
    # def make_frame(t):
    #     frame_idx = int(np.clip(np.round(t * mp4_fps), 0, num_frames - 1))
    #     col_dlatents = np.stack([dst_dlatents[0]])
    #     col_dlatents[:, style_ranges[0]] = src_dlatents[frame_idx, style_ranges[0]]
    #     col_images = Gs.components.synthesis.run(col_dlatents, randomize_noise=False, **synthesis_kwargs)
    #     return col_images[0]
    #
    # # Generate video.
    # import moviepy.editor
    # mp4_file = 'results/fine_%s.mp4' % (random_seed)
    # mp4_codec = 'libx264'
    # mp4_bitrate = '5M'
    #
    # video_clip = moviepy.editor.VideoClip(make_frame, duration=duration_sec)
    # video_clip.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)


def circular():
    tflib.init_tf()
    _G, _D, Gs = pickle.load(open(config.Model, "rb"))

    rnd = np.random
    latents_a = rnd.randn(1, Gs.input_shape[1])
    latents_b = rnd.randn(1, Gs.input_shape[1])
    latents_c = rnd.randn(1, Gs.input_shape[1])

    def circ_generator(latents_interpolate):
        radius = 40.0

        latents_axis_x = (latents_a - latents_b).flatten() / linalg.norm(latents_a - latents_b)
        latents_axis_y = (latents_a - latents_c).flatten() / linalg.norm(latents_a - latents_c)

        latents_x = math.sin(math.pi * 2.0 * latents_interpolate) * radius
        latents_y = math.cos(math.pi * 2.0 * latents_interpolate) * radius

        latents = latents_a + latents_x * latents_axis_x + latents_y * latents_axis_y
        return latents

    def mse(x, y):
        return (np.square(x - y)).mean()

    def generate_from_generator_adaptive(gen_func):
        max_step = 1.0
        current_pos = 0.0

        change_min = 10.0
        change_max = 11.0

        fmt = dict(func=tflib.convert_images_to_uint8, nchw_to_nhwc=True)

        current_latent = gen_func(current_pos)
        current_image = Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
        array_list = []

        video_length = 1.0
        while current_pos < video_length:
            array_list.append(current_image)

            lower = current_pos
            upper = current_pos + max_step
            current_pos = (upper + lower) / 2.0

            current_latent = gen_func(current_pos)
            current_image = images = \
            Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
            current_mse = mse(array_list[-1], current_image)

            while current_mse < change_min or current_mse > change_max:
                if current_mse < change_min:
                    lower = current_pos
                    current_pos = (upper + lower) / 2.0

                if current_mse > change_max:
                    upper = current_pos
                    current_pos = (upper + lower) / 2.0

                current_latent = gen_func(current_pos)
                current_image = images = \
                Gs.run(current_latent, None, truncation_psi=0.7, randomize_noise=False, output_transform=fmt)[0]
                current_mse = mse(array_list[-1], current_image)
            print("%s / %s : %s" % (current_pos, video_length, current_mse))
        return array_list

    frames = generate_from_generator_adaptive(circ_generator)
    frames = moviepy.editor.ImageSequenceClip(frames, fps=30)

    # Generate video.
    mp4_file = 'results/circular.mp4'
    mp4_codec = 'libx264'
    mp4_bitrate = '3M'
    mp4_fps = 20

    frames.write_videofile(mp4_file, fps=mp4_fps, codec=mp4_codec, bitrate=mp4_bitrate)


if __name__ == "__main__":
    main()
    circular()

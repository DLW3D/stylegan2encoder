# Copyright (c) 2019, NVIDIA CORPORATION. All rights reserved.
#
# This work is licensed under the Creative Commons Attribution-NonCommercial
# 4.0 International License. To view a copy of this license, visit
# http://creativecommons.org/licenses/by-nc/4.0/ or send a letter to
# Creative Commons, PO Box 1866, Mountain View, CA 94042, USA.

"""Global configuration."""

#----------------------------------------------------------------------------
# Paths.

result_dir = 'results'
data_dir = 'datasets'
cache_dir = 'cache'
dlatents_dir = 'latent_representations'
generated_dir = 'generated_images'
run_dir_ignore = ['results', 'datasets', 'cache', 'latent_representations', 'generated_images']

# experimental - replace Dense layers with TreeConnect
use_treeconnect = False
treeconnect_threshold = 1024
randomize_noise = False
allow_growth = True

Model = './models/2020-01-11-skylion-stylegan2-animeportraits-networksnapshot-024664.pkl'
# Model = './models/stylegan2-ffhq-config-f.pkl'
vgg16 = './models/vgg16_zhang_perceptual.pkl'

#----------------------------------------------------------------------------

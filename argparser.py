import argparse

def parse_arguments(*args):
    parser = argparse.ArgumentParser()

    ###############added options#######################################
    parser.add_argument('-lr', '--learning_rate', default=1e-3, type=float,
                        help='Learning rate for the generator')
    parser.add_argument('-lrd', '--learning_rate_D', default=1e-4, type=float,
                        help='Learning rate for the discriminator')
    parser.add_argument('-lrd_l', '--learning_rate_D_local', default=1e-4, type=float,
                        help='Learning rate for the discriminator')

    parser.add_argument('--gan', default='lsgan', type=str, choices=['dcgan', 'lsgan', 'wgan', 'improved wgan'],
                        help='dcgan|lsgan|wgan|improved wgan')  # todo wgan/improved wgan

    parser.add_argument('--model', default='scribbler', type=str, choices=['scribbler', 'texturegan', 'pix2pix','scribbler_dilate_128'],
                        help='scribbler|pix2pix')

    parser.add_argument('--num_epoch', default=100, type=int,
                        help='texture|scribbler')

    parser.add_argument('--visualize_every', default=10, type=int,
                        help='no. iteration to visualize the results')

    # all the weights ratio, might wanna make them sum to one
    parser.add_argument('--feature_weight', default=0, type=float,
                        help='weight ratio for feature loss')
    parser.add_argument('--global_pixel_weight_l', default=0, type=float,
                        help='weight ratio for pixel loss for l channel')
    parser.add_argument('--local_pixel_weight_l', default=1,  type=float,
                        help='pixel weight for local loss patch')
    parser.add_argument('--pixel_weight_ab', default=0, type=float,
                        help='weight ratio for pixel loss for ab channel')
    parser.add_argument('--pixel_weight_rgb', default=0, type=float,
                        help='weight ratio for pixel loss for ab channel')

    parser.add_argument('--discriminator_weight', default=0, type=float,
                        help='weight ratio for the discriminator loss')
    parser.add_argument('--discriminator_local_weight', default=0, type=float,
                        help='weight ratio for the discriminator loss')
    parser.add_argument('--style_weight', default=0, type=float,
                        help='weight ratio for the texture loss')

    # parser.add_argument('--gpu', default=[0], type=int, nargs='+',
    #                     help='List of GPU IDs to use')  # TODO support cpu
    parser.add_argument('--gpu', default=1, type=int, help="GPU ID")

    parser.add_argument('--display_port', default=7779, type=int,
                        help='port for displaying on visdom (need to match with visdom currently open port)')

    parser.add_argument('--data_path', default='/home/psangkloy3/training_handbags_pretrain/', type=str,
                        help='path to the data directory, expect train_skg, train_img, val_skg, val_img')

    parser.add_argument('--save_dir', default='/home/psangkloy3/test/', type=str,
                        help='path to save the model')

    parser.add_argument('--load_dir', default='/home/psangkloy3/test/', type=str,
                        help='path to save the model')

    parser.add_argument('--save_every', default=1000, type=int,
                        help='no. iteration to save the models')

    parser.add_argument('--load_epoch', default=-1, type=int,
                        help="The epoch number for the model to load")
    parser.add_argument('--load', default=-1, type=int,
                        help='load generator and discrminator from iteration n')
    parser.add_argument('--load_D', default=-1, type=int,
                        help='load discriminator from iteration n, priority over load')

    parser.add_argument('--image_size', default=128, type=int,
                        help='Training images size, after cropping')
    parser.add_argument('--resize_to', default=300, type=int,
                        help='Training images size, after cropping')
                        
    parser.add_argument('--resize_max', default=1, type=float,
                        help='max resize, ratio of the original image, max value is 1')
    parser.add_argument('--resize_min', default=0.6, type=float,
                        help='min resize, ratio of the original image, min value 0')
    parser.add_argument('--patch_size_min', default=20, type=int,
                        help='minumum texture patch size')
    parser.add_argument('--patch_size_max', default=40, type=int,
                        help='max texture patch size')

    parser.add_argument('--batch_size', default=32, type=int, help="Training batch size. MUST BE EVEN NUMBER")

    parser.add_argument('--num_input_texture_patch', default=2,type=int)
    parser.add_argument('--num_local_texture_patch', default=1,type=int)

    parser.add_argument('--color_space', default='lab', type=str, choices=['lab', 'rgb'],
                        help='lab|rgb')

    parser.add_argument('--threshold_D_max', default=0.8, type=int,
                        help='stop updating D when accuracy is over max')

    parser.add_argument('--content_layers', default='relu4_2', type=str,
                        help='Layer to attach content loss.')
    parser.add_argument('--style_layers', default='relu3_2, relu4_2', type=str,
                        help='Layer to attach content loss.')

    parser.add_argument('--use_segmentation_patch', default=True, type=bool,
                        help='whether or not to inject noise into the network')

    parser.add_argument('--input_texture_patch', default='dtd_texture', type=str,
                        choices=['original_image', 'dtd_texture'],
                        help='whether or not to inject noise into the network')
    
    parser.add_argument('--loss_texture', default='dtd_texture', type=str,
                        choices=['original_image', 'dtd_texture'],
                        help='where is the texture loss come from')
    
    parser.add_argument('--local_texture_size', default=50, type=int,
                        help='use local texture loss instead of global, set -1 to use global')
    
    parser.add_argument('--texture_discrminator_loss', default=True, type=bool,
                        help='adding discrminator for texture')
    
    ############################################################################
    ############################################################################
    ############Not Currently Using #################################################################
    parser.add_argument('--tv_weight', default=1, type=float,
                        help='weight ratio for total variation loss')

    parser.add_argument('--mode', default='texture', type=str, choices=['texture', 'scribbler'],
                        help='texture|scribbler')
    
    parser.add_argument('--visualize_mode', default='train', type=str, choices=['train', 'test'],
                        help='train|test')

    parser.add_argument('--crop', default='random', type=str, choices=['random', 'center'],
                        help='random|center')

    parser.add_argument('--contrast', default=True, type=bool,
                        help='randomly adjusting contrast on sketch')

    parser.add_argument('--occlude', default=False, type=bool,
                        help='randomly occlude part of the sketch')

    parser.add_argument('--checkpoints_path', default='data/', type=str,
                        help='output directory for results and models')

    parser.add_argument('--noise_gen', default=False, type=bool,
                        help='whether or not to inject noise into the network')

    parser.add_argument('--absolute_load', default='', type=str,
                        help='load saved generator model from absolute location')

    ##################################################################################################################################
    
    return parser.parse_args(*args)
    
        

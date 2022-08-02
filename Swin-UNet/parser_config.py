import argparse

def get_parser():
    remote_dir = '/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/data/'
    local_dir = './data/'

    parser = argparse.ArgumentParser()
    parser.add_argument('--root_path', type=str,
                        default=remote_dir, help='root dir for data')
    parser.add_argument('--dataset', type=str,
                        default='Spine', help='experiment_name')
    parser.add_argument('--num_classes', type=int,
                        default=1, help='output channel of network')
    parser.add_argument('--output_dir', type=str,
                        default='/home/pose3d/projs/STCN/UNet_Spine_Proj/UNet_Spine/output', help='output dir')
    parser.add_argument('--max_epochs', type=int,
                        default=100, help='maximum epoch number to train')
    parser.add_argument('--batch_size', type=int,
                        default=24, help='batch_size per gpu')
    parser.add_argument('--n_gpu', type=int, default=1, help='total gpu')
    parser.add_argument('--deterministic', type=int,  default=1,
                        help='whether use deterministic training')
    parser.add_argument('--base_lr', type=float,  default=3e-4,
                        help='segmentation network learning rate')
    parser.add_argument('--val_percent', type=float,
                        default=0.1, help='val_set percent')
    parser.add_argument('--img_size', type=int,
                        default=224, help='input patch size of network input')
    parser.add_argument('--seed', type=int,
                        default=1234, help='random seed')
    parser.add_argument('--cfg', type=str, default="./configs/swin_tiny_patch4_window7_224_lite.yaml",
                        metavar="FILE", help='path to config file', )
    parser.add_argument(
        "--opts",
        help="Modify config options by adding 'KEY VALUE' pairs.",
        default=None,
        nargs='+',
    )
    parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                        help='no: no cache, '
                        'full: cache all data, '
                        'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
    parser.add_argument('--resume', help='resume from checkpoint')
    parser.add_argument('--accumulation-steps', type=int,
                        help="gradient accumulation steps")
    parser.add_argument('--use-checkpoint', action='store_true',
                        help="whether to use gradient checkpointing to save memory")
    parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                        help='mixed precision opt level, if O0, no amp is used')
    parser.add_argument('--tag', help='tag of experiment')
    parser.add_argument('--eval', action='store_true',
                        help='Perform evaluation only')
    parser.add_argument('--throughput', action='store_true',
                        help='Test throughput only')

    args = parser.parse_args()
    return args
    
import os.path as osp
from collections import OrderedDict
import socket
import getpass
import pdb
machine_name = socket.gethostname()
username = getpass.getuser()

__all__ = ['parse_args']

def parse_args(parser):
    parser.add_argument('--data_root', default=osp.expanduser('/home/cli/hdd'), type=str)
    parser.add_argument('--save_path', default=osp.expanduser('/home/cli/exp_trn'), type=str)
    parser.add_argument('--phases', default=['train', 'test'], type=list)
    parser.add_argument('--test_interval', default=1, type=int)
    parser.add_argument('--width', default=1280, type=int)
    parser.add_argument('--height', default=720, type=int)

    args = parser.parse_args()
    args.data = osp.basename(osp.normpath(args.data_root))

    if 'sv800985lx' in machine_name and username == 'cli':  # TODO Titan XP @ 4
        args.data_root = '/home/cli/hdd'
    elif 'sv802661lx' in machine_name and username == 'cli':  # TODO RTX 6000 @ 3
        args.data_root = '/home/cli/hdd'
    elif 'sm120145' in machine_name and username == 'cli':  # TODO Titan P100 @ 8
        args.data_root = '/data1/hdd'
    elif 'sv802681lx' in machine_name and username == 'zxiao':
        args.data_root = '/home/zxiao/data/hdd'
    else:
        exit("INVALID USER: %s@%s" % (username, machine_name))

    if args.data =='hdd':
        args.data = 'HDD'
    args.class_index = list(data_info[args.data]['class_info'].keys())
    args.class_weight = list(data_info[args.data]['class_info'].values())
    args.intention_index = list(data_info[args.data]['intention_info'].keys())
    args.train_session_set = data_info[args.data]['train_session_set']
    args.test_session_set = data_info[args.data]['test_session_set']
    args.num_classes = len(args.class_index)
    return args

data_info = OrderedDict()
data_info['HDD'] = OrderedDict()

data_info['HDD']['intention_info'] = OrderedDict([
    ('background',               1.0),
    ('intersection passing',     1.0),
    ('left turn',                1.0),
    ('right turn',               1.0),
    ('left lane change',         1.0),
    ('right lane change',        1.0),
    ('left lane branch',         1.0),
    ('right lane branch',        1.0),
    ('crosswalk passing',        1.0),
    ('railroad passing',         1.0),
    ('merge',                    1.0),
    ('U-turn',                   1.0),
])

data_info['HDD']['class_info'] = OrderedDict([
    ('go',               1.0),
    ('stop',     1.0),
])
'''
data_info['HDD']['train_session_set'] = ['201704130952']
data_info['HDD']['test_session_set'] = ['201704130952']
'''
data_info['HDD']['train_session_set'] = [
    '201702271017', '201702271123', '201702271136', '201702271438',
    '201702271632', '201702281017', '201702281511', '201702281709',
    '201703011016', '201703061033', '201703061107', '201703061323',
    '201703061353', '201703061418', '201703061429', '201703061456',
    '201703061519', '201703061541', '201703061606', '201703061635',
    '201703061700', '201703061725', '201703080946', '201703081008',
    '201703081055', '201703081152', '201703081407', '201703081437',
    '201703081509', '201703081549', '201703081617', '201703081653',
    '201703081723', '201703081749', '201704101354', '201704101504',
    '201704101624', '201704101658', '201704110943', '201704111011',
    '201704111041', '201704111138', '201704111202', '201704111315',
    '201704111335', '201704111402', '201704111412', '201704111540',
    '201706061021', '201706070945', '201706071021', '201706071319',
    '201706071458', '201706071518', '201706071532', '201706071602',
    '201706071620', '201706071630', '201706071658', '201706071735',
    '201706071752', '201706080945', '201706081335', '201706081445',
    '201706081626', '201706081707', '201706130952', '201706131127',
    '201706131318', '201706141033', '201706141147', '201706141538',
    '201706141720', '201706141819', '201709200946', '201709201027',
    '201709201221', '201709201319', '201709201530', '201709201605',
    '201709201700', '201709210940', '201709211047', '201709211317',
    '201709211444', '201709211547', '201709220932', '201709221037',
    '201709221238', '201709221313', '201709221435', '201709221527',
    '201710031224', '201710031247', '201710031436', '201710040938',
    '201710060950', '201710061114', '201710061311', '201710061345',
]
data_info['HDD']['test_session_set'] = [
    '201704101118', '201704130952', '201704131020', '201704131047',
    '201704131123', '201704131537', '201704131634', '201704131655',
    '201704140944', '201704141033', '201704141055', '201704141117',
    '201704141145', '201704141243', '201704141420', '201704141608',
    '201704141639', '201704141725', '201704150933', '201704151035',
    '201704151103', '201704151140', '201704151315', '201704151347',
    '201704151502', '201706061140', '201706061309', '201706061536',
    '201706061647', '201706140912', '201710031458', '201710031645',
    '201710041102', '201710041209', '201710041351', '201710041448',
]


#data_info['HDD']['train_session_set'] = ['201704101118']
#data_info['HDD']['test_session_set'] = [ '201704101118']

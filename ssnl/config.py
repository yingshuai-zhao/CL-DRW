epochs = 100

# optimizer
lr = 1e-3
lr_INN = 1e-4
betas = (0.9, 0.999)
gamma = 0.5
weight_decay = 1e-5

# Parameters
a = 1
b = 0.01
c = 0.0001


train_batchsize = 8
train_batchsize_INN = 8
valid_batchsize = 8

# data path
TRAIN_ORI_PATH = './data/Original/train'
VALID_ORI_PATH = './data/Original/test'


TRAIN_QQ_PATH = './data/QQDownload/train'
TRAIN_FB_PATH = './data/FacebookDownload/train'
TRAIN_WC_PATH = './data/WeChatDownload/train'

VALID_QQ_PATH = './data/QQDownload/test'
VALID_FB_PATH = './data/FacebookDownload/test'
VALID_WC_PATH = './data/WeChatDownload/test'

TRAIN_OSN_PATH = TRAIN_FB_PATH
VALID_OSN_PATH = VALID_FB_PATH


# TRAIN_ORI_PATH = '/home/zhaoyingshuai/OSN_Trans_Images/Original128'
# VALID_ORI_PATH = '/home/zhaoyingshuai/OSN_Trans_Images/Original128'

# TRAIN_OSN_PATH = '/home/zhaoyingshuai/OSN_Trans_Images/WC128'
# VALID_OSN_PATH = '/home/zhaoyingshuai/OSN_Trans_Images/WC128'


# Saving checkpoints:
# MODEL_PATH = 'experiments/Resformer/'
MODEL_PATH = 'experiments/SDN_WC/'
MODEL_PATH_INN = 'experiments/INN_WC/'
SAVE_freq = 1

suffix = 'sim_osn__013.pt'
train_continue = True

suffix_INN = ''
train_continue_INN = False

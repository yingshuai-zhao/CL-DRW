start_epoch = 151
epochs = 200
clamp = 2.0

# optimizer
lr = 1e-4
betas = (0.5, 0.999)
gamma = 0.5
weight_decay = 1e-5

# input settings
message_weight = 2
stego_weight = 20
message_length = 64


# Train:
batch_size_train = 48
batchsize_valid = 20
cropsize = 128


# # Data Path
# VAL_PATH = '/home/zhaoyingshuai/DIV2K/DIV2K_train_HR'
# VAL_PATH = '/home/zhaoyingshuai/DIV2K/DIV2K_valid_HR'


# Data Path
TRAIN_PATH = '/COCO/train/'
VAL_PATH = '/valid/'
TEST_PATH = './COCO-WC'


# Saving checkpoints:
MODEL_PATH = ''

suffix = ''
train_continue = True
diff = False







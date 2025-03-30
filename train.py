import torch.nn
from utils import *
import cfgs as c
from model.cldrw import CLDRW
from typical_noise_layers.jpeg import JpegTest
from cl_noiser import Noiser
from typical_noise_layers.diff_jpeg.jpeg import DiffJPEGCoding
from typical_noise_layers.resize import Resize
from typical_noise_layers.cropout import Cropout
from ssnl.models.models import Encoder_Unet, Decoder_Unet, Unet_OSN_DB, Unet_OSN, Restormer


device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")

loss_mse = torch.nn.MSELoss(reduce=True).to(device)

cldrw = CLDRW(c.cropsize, c.message_length)
cldrw.to(device)
params_trainable = (list(filter(lambda p: p.requires_grad, cldrw.parameters())))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

if c.train_continue: 
    load(cldrw, c.MODEL_PATH + c.suffix)

# load data
trainloader, validloader = get_data_loaders(c.TRAIN_PATH, c.VAL_PATH, c.batch_size_train, c.batchsize_valid, c.cropsize)


setup_logger('train', 'logging', 'train_', level=logging.INFO, screen=True, tofile=True)
logger_train = logging.getLogger('train')

noise_curriculum = Noiser()

noise_osn = Restormer().to(device)
checkpoint = torch.load('./weights/sim_fb_92.pt', map_location=torch.device('cpu'))
noise_osn.load_state_dict(checkpoint['net'])
noise_osn.eval()
noise_osn_jpeg = DiffJPEGCoding(40.0, ste=True)

test_noise_curriculum = Noiser()
test_jpeg = JpegTest(50)


for i_epoch in range(c.start_epoch, c.epochs):

    loss_history = []
    stego_loss_history = []
    message_loss_history = []
    stego_psnr_history = []
    error_history = []

    #################
    #     train:    #
    #################

    message_weight = max(c.message_weight, 20/i_epoch)

    if i_epoch <= 20:
        stego_weight = 1
    else:
        stego_weight = min(c.stego_weight, (i_epoch-20)/2)

    print('epoch ', i_epoch, ', stego_weight: ', stego_weight, ', message_weight', message_weight)

    cldrw.train()
    for idx_batch, [cover_img, _] in enumerate(trainloader):
        cover_img = cover_img.to(device)

        message = torch.Tensor(np.random.choice([-0.5, 0.5], (cover_img.shape[0], c.message_length))).to(device)
        input_data = [cover_img, message]

        #################
        #    forward:   #
        #################

        stego_img, left_noise = cldrw(input_data)


        # stage 1&2
        # stego_noise_img = noise_curriculum([stego_img.clone(),  cover_img.clone()], i_epoch)[0]
        
        # stage 3
        if idx_batch <= len(trainloader)*0.3:
            stego_noise_img = noise_osn(stego_img.clone())
        elif idx_batch <= len(trainloader)*0.8:
            stego_noise_img = noise_osn_jpeg(noise_osn(stego_img.clone()))
        else:
            stego_noise_img = noise_curriculum([stego_img.clone(),  cover_img.clone()], i_epoch)[0]



        #################
        #   backward:   #
        ################

        guass_noise = torch.zeros(left_noise.shape).to(device)
        output_data = [stego_noise_img, guass_noise]
        re_img, re_message = cldrw(output_data, rev=True)

        stego_loss = loss_mse(stego_img, cover_img)
        message_loss = loss_mse(re_message, message)

        total_loss = message_weight * message_loss + stego_weight * stego_loss
        total_loss.backward()

        optim.step()
        optim.zero_grad()

        psnr_temp_stego = psnr(cover_img, stego_img, 255)

        error_rate = decoded_message_error_rate_batch(message, re_message)

        loss_history.append([total_loss.item(), 0.])
        stego_loss_history.append([stego_loss.item(), 0.])
        message_loss_history.append([message_loss.item(), 0.])
        stego_psnr_history.append([psnr_temp_stego, 0.])
        error_history.append([error_rate, 0.])

    epoch_losses = np.mean(np.array(loss_history), axis=0)
    stego_epoch_losses = np.mean(np.array(stego_loss_history), axis=0)
    message_epoch_losses = np.mean(np.array(message_loss_history), axis=0)
    stego_psnr = np.mean(np.array(stego_psnr_history), axis=0)
    error = np.mean(np.array(error_history), axis=0)

    epoch_losses[1] = np.log10(optim.param_groups[0]['lr'])

    logger_train.info(f"Learning rate: {optim.param_groups[0]['lr']}")
    logger_train.info(
        f"Train epoch {i_epoch}:   "
        f'Loss: {epoch_losses[0].item():.6f} | '
        f'Stego_Loss: {stego_epoch_losses[0].item():.6f} | '
        f'Message_Loss: {message_epoch_losses[0].item():.6f} | '
        f'Stego_Psnr: {stego_psnr[0].item():.6f} |'
        f'Error:{1 - error[0].item():.6f} |'
    )

    #################
    #     val:      #
    #################
    with torch.no_grad():
        stego_psnr_history = []
        error_history = []
        error_history_ = []
        error_history_2 = []

        cldrw.eval()
        for test_cover_img, _ in validloader:
            test_cover_img = test_cover_img.to(device)

            test_message = torch.Tensor(np.random.choice([-0.5, 0.5], (test_cover_img.shape[0], c.message_length))).to(device)

            test_input_data = [test_cover_img, test_message]

            #################
            #    forward:   #
            #################

            test_stego_img, test_left_noise = cldrw(test_input_data)

            test_stego_noise_img = test_noise_curriculum([test_stego_img.clone(), test_cover_img.clone()], 1000)[0]
            test_stego_noise_img_ = test_jpeg(test_stego_img.clone())
            test_stego_noise_img_2 = noise_osn_jpeg(noise_osn(test_stego_img.clone()))


            #################
            #   backward:   #
            #################

            test_z_guass_noise = torch.zeros(test_left_noise.shape).to(device)
            test_z_guass_noise_ = torch.zeros(test_left_noise.shape).to(device)
            test_z_guass_noise_2 = torch.zeros(test_left_noise.shape).to(device)
            test_z_guass_noise_3 = torch.zeros(test_left_noise.shape).to(device)

            test_output_data = [test_stego_noise_img, test_z_guass_noise]
            test_output_data_ = [test_stego_noise_img_, test_z_guass_noise_]
            test_output_data_2 = [test_stego_noise_img_2, test_z_guass_noise_2]

            test_re_img, test_re_message = cldrw(test_output_data, rev=True)
            test_re_img_, test_re_message_ = cldrw(test_output_data_, rev=True)
            test_re_img_2, test_re_message_2 = cldrw(test_output_data_2, rev=True)


            psnr_temp_stego = psnr(test_cover_img, test_stego_img, 255)
            # psnr_temp_recover = psnr(test_cover_img, test_re_img, 255)
            # psnr_temp_recover_ = psnr(test_cover_img, test_re_img_, 255)

            error_rate = decoded_message_error_rate_batch(test_message, test_re_message)
            error_rate_ = decoded_message_error_rate_batch(test_message, test_re_message_)
            error_rate_2 = decoded_message_error_rate_batch(test_message, test_re_message_2)

            stego_psnr_history.append(psnr_temp_stego)
            error_history.append(error_rate)
            error_history_.append(error_rate_)
            error_history_2.append(error_rate_2)


        logger_train.info(
            f"TEST:   "
            f'PSNR_STEGO: {np.mean(stego_psnr_history):.6f} | '
            f'Typical_Noises: {1 - np.mean(error_history):.6f} | '
            f'Jpeg: {1 - np.mean(error_history_):.6f} | '
            f'SSNL: {1 - np.mean(error_history_2):.6f} | '
        )

    torch.save({'opt': optim.state_dict(),
                'net': cldrw.state_dict()},
                c.MODEL_PATH + 'cldrw_' + str(np.mean(stego_psnr_history)) + '_%.3i' % i_epoch + '.pt')

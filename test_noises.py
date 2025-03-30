import torch.nn
from utils import *
from model.cldrw import CLDRW
from typical_noise_layers.jpeg import JpegTest
from typical_noise_layers.resize import Resize
import kornia
from typical_noise_layers.salt_and_pepper import Salt_and_Pepper
from typical_noise_layers.dropout import Dropout
from typical_noise_layers.Gaussian_noise import Gaussian_Noise
import cfgs as c


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def load(model, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)

cldrw = CLDRW(c.cropsize, c.message_length)
cldrw.to(device)

# load data
trainloader, validloader = get_data_loaders(c.TRAIN_PATH, c.VAL_PATH, c.batch_size_train, c.batchsize_valid, c.cropsize)


# # without ssnl
# load(cldrw, 'experiments/WC/cldrw_42.602657_096.pt')

load(cldrw, 'experiments/WC/cldrw_42.431675_180.pt')

noises = [Gaussian_Noise(0, 1), Salt_and_Pepper(0.05), Resize([0.5, 0.5]), Dropout([0.3, 0.3])]
test_noise_layer_old = noises[0]
test_noise_layer_new = noises[1]
test_noise_layer_new2 = noises[2]
test_noise_layer_new3 = noises[3]
test_noise_layer_new4 = JpegTest(50)


with torch.no_grad():
    stego_psnr_history = []
    stego_ssim_history = []
    error_history_id = []
    error_history = []
    error_history_ = []
    error_history_2 = []
    error_history_3 = []
    error_history_4 = []

    cldrw.eval()
    for test_cover_img, _ in validloader:
        test_cover_img = test_cover_img.to(device)

        test_message = torch.Tensor(np.random.choice([-0.5, 0.5], (test_cover_img.shape[0], c.message_length))).to(device)

        test_input_data = [test_cover_img, test_message]

        #################
        #    forward:   #
        #################

        test_stego_img, test_left_noise = cldrw(test_input_data)


        # test_stego_noise_img = test_stego_img.clone()
        test_stego_noise_img = test_noise_layer_old([test_stego_img.clone(), test_cover_img.clone()])[0]
        test_stego_noise_img_ = test_noise_layer_new([test_stego_img.clone(), test_cover_img.clone()])[0]
        test_stego_noise_img_2 = test_noise_layer_new2([test_stego_img.clone(), test_cover_img.clone()])[0]
        test_stego_noise_img_3 = test_noise_layer_new3([test_stego_img.clone(), test_cover_img.clone()])[0]
        test_stego_noise_img_4 = test_noise_layer_new4(test_stego_img.clone())

        #################
        #   backward:   #
        #################

        test_z_guass = torch.zeros(test_left_noise.shape).to(device)
        test_z_guass_noise = torch.zeros(test_left_noise.shape).to(device)
        test_z_guass_noise_ = torch.zeros(test_left_noise.shape).to(device)
        test_z_guass_noise_2 = torch.zeros(test_left_noise.shape).to(device)
        test_z_guass_noise_3 = torch.zeros(test_left_noise.shape).to(device)
        test_z_guass_noise_4 = torch.zeros(test_left_noise.shape).to(device)


        test_output = [test_stego_img.clone(), test_z_guass]
        test_output_data = [test_stego_noise_img, test_z_guass_noise]
        test_output_data_ = [test_stego_noise_img_, test_z_guass_noise_]
        test_output_data_2 = [test_stego_noise_img_2, test_z_guass_noise_2]
        test_output_data_3 = [test_stego_noise_img_3, test_z_guass_noise_3]
        test_output_data_4 = [test_stego_noise_img_4, test_z_guass_noise_4]

        test_re, test_re_m = cldrw(test_output, rev=True)
        test_re_img, test_re_message = cldrw(test_output_data, rev=True)
        test_re_img_, test_re_message_ = cldrw(test_output_data_, rev=True)
        test_re_img_2, test_re_message_2 = cldrw(test_output_data_2, rev=True)
        test_re_img_3, test_re_message_3 = cldrw(test_output_data_3, rev=True)
        test_re_img_4, test_re_message_4 = cldrw(test_output_data_4, rev=True)


        psnr_temp_stego = psnr(test_cover_img, test_stego_img, 255)
        ssim = kornia.losses.ssim_loss(test_cover_img, test_stego_img, 5)
        # psnr_temp_recover = psnr(test_cover_img, test_re_img, 255)
        # psnr_temp_recover_ = psnr(test_cover_img, test_re_img_, 255)

        error_rate_i = decoded_message_error_rate_batch(test_message, test_re_m)
        error_rate = decoded_message_error_rate_batch(test_message, test_re_message)
        error_rate_ = decoded_message_error_rate_batch(test_message, test_re_message_)
        error_rate_2 = decoded_message_error_rate_batch(test_message, test_re_message_2)
        error_rate_3 = decoded_message_error_rate_batch(test_message, test_re_message_3)
        error_rate_4 = decoded_message_error_rate_batch(test_message, test_re_message_4)

        stego_psnr_history.append(psnr_temp_stego)
        stego_ssim_history.append(1-ssim.item())
        error_history_id.append(error_rate_i)
        error_history.append(error_rate)
        error_history_.append(error_rate_)
        error_history_2.append(error_rate_2)
        error_history_3.append(error_rate_3)
        error_history_4.append(error_rate_4)
        
    print(
        f"TEST:   "
        f'PSNR_STEGO: {np.mean(stego_psnr_history):.6f} | '
        f'SSIM_STEGO: {np.mean(stego_ssim_history):.6f} | '
        f'Accuracy_id: {1 - np.mean(error_history_id):.6f} | '
        f'Accuracy_old: {1 - np.mean(error_history):.6f} | '
        f'Accuracy_new: {1 - np.mean(error_history_):.6f} | '
        f'Accuracy_new2: {1 - np.mean(error_history_2):.6f} | '
        f'Accuracy_new3: {1 - np.mean(error_history_3):.6f} | '
        f'Accuracy_new4: {1 - np.mean(error_history_4):.6f} | '
    )


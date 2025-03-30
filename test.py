import torch.nn
from utils import *
from model.cldrw import CLDRW
import kornia
import cfgs as c


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

def load(model, name):
    state_dicts = torch.load(name)
    network_state_dict = {k: v for k, v in state_dicts['net'].items() if 'tmp_var' not in k}
    model.load_state_dict(network_state_dict)

cldrw = CLDRW(c.cropsize, c.message_length)
cldrw.to(device)
params_trainable = (list(filter(lambda p: p.requires_grad, cldrw.parameters())))

optim = torch.optim.Adam(params_trainable, lr=c.lr, betas=c.betas, eps=1e-6, weight_decay=c.weight_decay)

# # without ssnl
# load(cldrw, 'experiments/WC/cldrw_42.602657_096.pt')

# # without ssnl, but jpeg(40)
# load(cldrw, 'experiments/WC/cldrw_42.49121_110.pt')

# load(cldrw, 'experiments/WC/cldrw_42.296707_144.pt')

load(cldrw, 'experiments/WC/cldrw_42.431675_180.pt')

#################
#     val:      #
#################
with torch.no_grad():
    cldrw.eval()
    def encode():
        # load data
        trainloader, validloader = get_data_loaders(c.TRAIN_PATH, c.VAL_PATH, c.batch_size_train, 1, c.cropsize)
        error_history = []
        num = 1
        for test_cover_img, _ in validloader:
            np.random.seed(114)
            test_cover_img = test_cover_img.to(device)

            test_message = torch.Tensor(np.random.choice([-0.5, 0.5], (1, c.message_length))).to(device)
            test_message = test_message.repeat(test_cover_img.shape[0], 1)

            test_input_data = [test_cover_img, test_message]

            test_stego_img, test_left_noise = cldrw(test_input_data)

            psnr_temp_stego = psnr(test_cover_img, test_stego_img, 255)
            ssim = kornia.losses.ssim_loss(test_cover_img, test_stego_img, 5)

            test_z_guass_noise = torch.zeros([test_stego_img.shape[0], c.message_length]).to(device)
        
            output_data = [test_stego_img, test_z_guass_noise]
            re_img, re_message = cldrw(output_data, rev=True)


            error_history.append(psnr_temp_stego)

            num = save_images(test_stego_img, './real_test', num=num)

            print(psnr_temp_stego)

            
            # print(1-ssim.item())

            
            # num = save_images(test_cover_img, './cover', num=num)
            
            # save_valid_images(test_cover_img, test_stego_img, 8, num, './test_pics')
            # num += 1

            if num >= 10:
                break

        print(f'Accuracy_old: {np.mean(error_history):.6f}')

        return test_message


        #################
        #   backward:   #
        #################
    def decode():
        err = []

        realtestloader = get_data_loaders_test('./a', 1, 128)

        for test_stego_noise_img, _ in realtestloader:
            test_stego_noise_img = test_stego_noise_img.to(device)

            np.random.seed(114)

            test_message_target = torch.Tensor(np.random.choice([-0.5, 0.5], (1, c.message_length))).to(device)
            test_message_target = test_message_target.repeat(test_stego_noise_img.shape[0], 1)

            test_z_guass_noise = torch.zeros([test_stego_noise_img.shape[0], c.message_length]).to(device)
        
            output_data = [test_stego_noise_img, test_z_guass_noise]
            re_img, re_message = cldrw(output_data, rev=True)

            error_rate = decoded_message_error_rate_batch(test_message_target, re_message)

            print(1-error_rate)

            err.append(error_rate)

        print(1-np.mean(err))

decode()

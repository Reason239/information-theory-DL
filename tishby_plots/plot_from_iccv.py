from plots_refactored import plot_all_epochs
import numpy as np

def iccv_to_tishby(data_dir, num_layers):
    I_XL_arrays = []
    I_LY_arrays = []
    for layer in range(num_layers):
        xl_path = data_dir + f'\\Info_X{layer + 1}.npy'
        ly_path = data_dir + f'\\Info_Y{layer + 1}.npy'
        I_XL_arrays.append(np.load(xl_path))
        I_LY_arrays.append(np.load(ly_path))
    I_XL_array = np.dstack(I_XL_arrays).squeeze()
    I_LY_array = np.dstack(I_LY_arrays).squeeze()
    return I_XL_array, I_LY_array

if __name__ == '__main__':
    # data_dir = r'C:\Users\kirik\PycharmProjects\2019_paper_code\VanillaInfoPlain\KT_testing\big1\Instance_0\InfoPlain0.001\VanillaMLP_relu'
    data_dir = r'C:\Users\kirik\Курсач\information-theory-DL\tishby_plots\iccv output\colab1\Gamma_0_Activation_relu'
    I_XL_array, I_LY_array = iccv_to_tishby(data_dir=data_dir, num_layers=3)
    epochs_inds = np.load(data_dir + r'\InfoStep.npy')[:len(I_XL_array)]
    plot_all_epochs(I_XL_array, I_LY_array, epochs_inds, title_str='big1', save_name='big1', fig_size=(10, 10))

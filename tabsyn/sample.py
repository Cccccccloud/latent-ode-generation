import torch

import argparse
import warnings
import time

from tabsyn.model import MLPDiffusion, Model
from tabsyn.latent_utils import get_input_generate, recover_data, split_num_cat_target
from tabsyn.diffusion_utils import sample

warnings.filterwarnings('ignore')


def main(args):
    dataname = args.dataname
    device = args.device
    steps = args.steps
    save_path = args.save_path

    train_z, _, _, ckpt_path, info, num_inverse, cat_inverse = get_input_generate(args)
    in_dim = train_z.shape[1] 

    mean = train_z.mean(0)

    denoise_fn = MLPDiffusion(in_dim, 1024).to(device)
    
    model = Model(denoise_fn = denoise_fn, hid_dim = train_z.shape[1]).to(device)

    model.load_state_dict(torch.load(f'{ckpt_path}/model.pt'))

    
    
    '''
        Generating samples    
    '''
    start_time = time.time()

    if hasattr(args, 'num_samples') and args.num_samples is not None:
        num_samples = args.num_samples
    else:
        num_samples = train_z.shape[0]
        
    sample_dim = in_dim
    chunk_size = 256
    
    import numpy as np
    syn_num_list, syn_cat_list, syn_target_list = [], [],[]

    print(f"Generating {num_samples} samples in chunks of {chunk_size}...")

    with torch.no_grad():
        for i in range(0, num_samples, chunk_size):
            curr_size = min(chunk_size, num_samples - i)
            
            x_next_chunk = sample(model.denoise_fn_D, curr_size, sample_dim)
            x_next_chunk = x_next_chunk * 2 + mean.to(device)
            syn_data_chunk = x_next_chunk.float().cpu().numpy()
            
            syn_num_c, syn_cat_c, syn_target_c = split_num_cat_target(
                syn_data_chunk, info, num_inverse, cat_inverse, args.device
            )
            
            if syn_num_c is not None: syn_num_list.append(syn_num_c)
            if syn_cat_c is not None: syn_cat_list.append(syn_cat_c)
            if syn_target_c is not None: syn_target_list.append(syn_target_c)

    syn_num = np.concatenate(syn_num_list, axis=0) if len(syn_num_list) > 0 else None
    syn_cat = np.concatenate(syn_cat_list, axis=0) if len(syn_cat_list) > 0 else None
    syn_target = np.concatenate(syn_target_list, axis=0) if len(syn_target_list) > 0 else None

    syn_df = recover_data(syn_num, syn_cat, syn_target, info)


    idx_name_mapping = info['idx_name_mapping']
    idx_name_mapping = {int(key): value for key, value in idx_name_mapping.items()}

    syn_df.rename(columns = idx_name_mapping, inplace=True)
    syn_df.to_csv(save_path, index = False)
    
    end_time = time.time()
    print('Time:', end_time - start_time)

    print('Saving sampled data to {}'.format(save_path))

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Generation')

    parser.add_argument('--dataname', type=str, default='adult', help='Name of dataset.')
    parser.add_argument('--gpu', type=int, default=0, help='GPU index.')
    parser.add_argument('--epoch', type=int, default=None, help='Epoch.')
    parser.add_argument('--steps', type=int, default=None, help='Number of function evaluations.')

    args = parser.parse_args()

    # check cuda
    if args.gpu != -1 and torch.cuda.is_available():
        args.device = f'cuda:{args.gpu}'
    else:
        args.device = 'cpu'
        

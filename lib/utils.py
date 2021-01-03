from datetime import datetime
import numpy as np
import torch


def save_model_with_meta(model, file, optimizer, additional_info):
    dict_to_save = {'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'date': str(datetime.now()), 'additional_info': {}}
    dict_to_save['additional_info'].update(additional_info)
    torch.save(dict_to_save, file)


def load_model_with_meta(model, path, device, verbose=False):
    loaded_torch = torch.load(path, map_location=device)

    if 'model_state_dict' in loaded_torch:
        model.load_state_dict(loaded_torch['model_state_dict'])
    else:
        model.load_state_dict(loaded_torch)

    model.eval()
    model.to(device)

    if verbose:
        if 'model_state_dict' in loaded_torch:
            print('Model Info:')
            for key, item in loaded_torch.items():
                if key == 'model_state_dict' or key == 'optimizer_state_dict':
                    continue
                print('%s: %s' % (key, item))


def init_loader_seed(worker_id):
    """
    Used to avoid multiple workers having the same seed for random events
    :param worker_id: Matches the torch interface
    """
    np.random.seed(torch.initial_seed() % 2 ** 32)

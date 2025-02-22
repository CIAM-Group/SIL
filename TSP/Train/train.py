##########################################################################################
# Machine Environment Config

DEBUG_MODE = False
USE_CUDA = not DEBUG_MODE
CUDA_DEVICE_NUM = 0

##########################################################################################
# Path Config

import os
import sys

os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, "..")  # for problem_def
sys.path.insert(0, "../..")  # for utils

##########################################################################################
# import

import logging
from utils.utils import create_logger, copy_all_src
from TSP.Train.TSPTrainer import TSPTester as Tester

##########################################################################################
# parameters
b = os.path.abspath(".").replace('\\', '/')
b2 = os.path.abspath("..").replace('\\', '/')

env_params = {
    'mode': 'test',
    'test_in_tsplib': False,
    'tsplib_path': b + '/data/ALL_tsp/coordinate/TSPlib65.txt',
    'data_path':
        'temp_repeaired_data.txt',
    'budget': 100,
    'data_path_pt': [b + '/repaired_data', '/repeaired_data.pt', '/repeaired_data_solution.pt'],
    'PRC': True,  # True: use parallel reconstruction; False: use single segment reconstruction
    'max_subtour_length': 1000,
    'repair_max_sub_length': 1000,
    'RI_initial': True
}

model_params = {
    'mode': 'test',
    'embedding_dim': 128,
    'sqrt_embedding_dim': 128 ** (1 / 2),
    'encoder_layer_num': 6,
    'qkv_dim': 16,
    'head_num': 8,
    'logit_clipping': 10,
    'ff_hidden_dim': 512,
    'k_nearest': False,
    'k_nearest_num': 1000
}

optimizer_params = {
    'optimizer': {
        'lr': 1e-4,
    },
    'scheduler': {
        'milestones': [1 * i for i in range(1, 300)],
        'gamma': 0.97}
}

tester_params = {
    'use_cuda': USE_CUDA,
    'cuda_device_num': CUDA_DEVICE_NUM,
    'model_load': {
        'path': None,
        'epoch': None,
    },
    'data_path': b2 + "/data/validation_set/validation_TSP1000_n128.txt",
    'episodes': 4,
    'batch_size': 4,
}

trainer_params = {
    'episodes': [20000],  # 65
    'problem_sizes': [1000],
    'repair_batch_size': [256],
    'train_batch_size': [256],
    'improve_iterations': 100,
    'epoch_itervel': 20,
    'first_time_repair': True,
    'repair_before_train': True,
    'epochs': 3000,
    'logging': {
        'model_save_interval': 1,
        'img_save_interval': 3000,
        'log_image_params_1': {
            'json_foldername': 'log_image_style',
            'filename': 'style_tsp_100.json'
        },
        'log_image_params_2': {
            'json_foldername': 'log_image_style',
            'filename': 'style_loss_1.json'
        },
    },
}

logger_params = {
    'log_file': {
        'desc': 'train_by_repair',
        'filename': 'log.txt'
    }
}


##########################################################################################
# main

def main(epoch, path, problem_sizes, eposides, train_batch_sizes, repair_batch_size,
         improve_iterations, budget, max_subtour_length, repair_max_sub_length,
         first_time_repair, repair_before_train, epoch_itervel, validation_sets):
    logger_params['log_file']['desc'] = f'train_by_SIL_TSP{problem_sizes[0]}'

    tester_params['model_load'] = {
        'path': path,
        'epoch': epoch,
    }
    trainer_params['episodes'] = eposides
    trainer_params['problem_sizes'] = problem_sizes
    trainer_params['repair_batch_size'] = repair_batch_size

    trainer_params['train_batch_size'] = train_batch_sizes
    trainer_params['first_time_repair'] = first_time_repair
    trainer_params['repair_before_train'] = repair_before_train
    trainer_params['improve_iterations'] = improve_iterations
    trainer_params['epoch_itervel'] = epoch_itervel

    env_params['budget'] = budget
    env_params['max_subtour_length'] = max_subtour_length
    env_params['repair_max_sub_length'] = repair_max_sub_length
    env_params['data_path_pt'] = [b + f'/repaired_data{problem_sizes[0]}',
                                  f'/repaired_data{problem_sizes[0]}.pt',
                                  f'/repaired_data_solution{problem_sizes[0]}.pt']

    # Optionmal. When using greedy search for validation on instances of size 50000/100000,
    # it takes a lot of time. To avoid this, we limit the number of input nodes to 10000.

    if problem_sizes[0] >= 10000:
        model_params['k_nearest_num'] = 10000

    tester_params['data_path'] = validation_sets[problem_sizes[0]][0]
    tester_params['episodes'] = validation_sets[problem_sizes[0]][1]
    tester_params['batch_size'] = validation_sets[problem_sizes[0]][2]

    create_logger(**logger_params)
    _print_config()

    tester = Tester(env_params=env_params,
                    model_params=model_params,
                    tester_params=tester_params,
                    trainer_params=trainer_params,
                    optimizer_params=optimizer_params)

    copy_all_src(tester.result_folder)

    tester.run()
    return


def _print_config():
    logger = logging.getLogger('root')
    logger.info('DEBUG_MODE: {}'.format(DEBUG_MODE))
    logger.info('USE_CUDA: {}, CUDA_DEVICE_NUM: {}'.format(USE_CUDA, CUDA_DEVICE_NUM))
    [logger.info(g_key + "{}".format(globals()[g_key])) for g_key in globals().keys() if g_key.endswith('params')]


##########################################################################################

if __name__ == "__main__":
    paras = {
        # meaning:
        # problem_size: [dataset_size, train_batch_size, repair_batch_size, budget, improve_iterations]
        # 100: [1000000, 1024, 1024, 5], pre-training on scale 100 and then training further on scale 1000 may lead to better performance.
        1000: [20000, 256, 256, 8],
        5000: [200, 32, 32, 8],
        10000: [200, 32, 32, 8],
        50000: [100, 32, 32, 8],
        100000: [100, 16, 16, 8],
    }

    validation_sets = {
        # 100: [b2+"/data/validation_set/validation_TSP100_n10000.txt", 10000, 1000],
        1000: [b2 + "/data/validation_set/validation_TSP1000_n128.txt", 128, 32],
        5000: [b2 + "/data/validation_set/validation_TSP5000_n128.txt", 16, 16],
        10000: [b2 + "/data/validation_set/validation_TSP10000_n16.txt", 16, 16],
        50000: [b2 + "/data/validation_set/validation_TSP50000_n4.txt", 4, 4],
        100000: [b2 + "/data/validation_set/validation_TSP100000_n4.txt", 4, 4],
    }

    problem_sizes = [1000]

    epoch_itervel = 20
    max_subtour_length = 1000
    repair_max_sub_length = 1000
    budget = 100

    ''' 1. RI_initial = True:  Train the model from scratch, the pre-trained model is not required. 
                               The Pseudo-labels are generated by random insertion, a simple heuristic method.
        2. RI_initial = False: Continue training the provided model. For example, a model trained on CVRP/TSP1K continues to be trained on CVRP/TSP10K. 
                               Parameters should be changed accordingly. Alternatively, continue the interrupted training process. 
                               Or continue the interrupted training process. '''
    RI_initial = True
    path = None
    epoch = None

    eposides = [paras[problem_sizes[0]][0]]
    train_batch_sizes = [paras[problem_sizes[0]][1]]
    repair_batch_size = [paras[problem_sizes[0]][2]]
    improve_iterations = paras[problem_sizes[0]][3]

    ''' 1. if no pre-trained model, and no labels: 
            first_time_repair=True, repair_before_train=True;
        2. if pre-trained model is obtained, and no labels / labels needed to be refined,
            first_time_repair=False, repair_before_train=True;
        3. if no pre-trained model, and labels are obtained: 
            first_time_repair=True, repair_before_train=False;
        4. if both pre-trained model and labels are obtained, and the labels are not needed to be refined,
           first_time_repair=False, repair_before_train=False. '''

    first_time_repair = True
    repair_before_train = True

    main(epoch, path, problem_sizes, eposides, train_batch_sizes, repair_batch_size,
         improve_iterations, budget, max_subtour_length, repair_max_sub_length,
         first_time_repair, repair_before_train, epoch_itervel, validation_sets)

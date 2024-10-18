import pandas as pd
import yaml
import argparse
import torch
from model import YNet

CONFIG_FILE_PATH = 'ynet_additional_files/config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATASET_NAME = 'sdd'

TEST_DATA_PATH = 'test/scene/plat.pkl'
TEST_IMAGE_PATH = 'test/scene'  # only needed for YNet, PECNet ignores this value
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

ROUNDS = 1  # Y-net is stochastic. How often to evaluate the whole dataset
BATCH_SIZE = 20

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

df_test = pd.read_pickle(TEST_DATA_PATH)
df_test['sceneId'] = 'plat'
df_test['metaId'] = 0
column_mapping = {'X': 'x', 'Y': 'y'}
df_test = df_test.rename(columns=column_mapping)

#TODO: 1.测试分割 2.可视化

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params)
model.load(f'ynet_additional_files/pretrained_models/{experiment_name}_weights.pt')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


model.evaluate(df_test, params, image_path=TEST_IMAGE_PATH, batch_size=BATCH_SIZE, rounds=ROUNDS, 
               num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=device, dataset_name=DATASET_NAME)


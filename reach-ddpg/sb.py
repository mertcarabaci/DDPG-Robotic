from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise, OrnsteinUhlenbeckActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import os
from environment import ReachEnv
import time
import matplotlib.pyplot as plt
import sys
import numpy as np
import math
import random
from datetime import datetime
import argparse

models_dir = f"models/{int(time.time())}/"
logdir = f"logs/{int(time.time())}/"
print(f"logdir: {logdir}")
date_and_time = datetime.now().strftime("%Y%m%d-%H%M%S")

if not os.path.exists(models_dir):
	os.makedirs(models_dir)
        
if not os.path.exists(logdir):
	os.makedirs(logdir)

EPISODES = 200 #default = 1000
EPISODE_LENGTH = 300

iters = 0


#H sütun
#- injekt nonun parantez içindeki harfi ile production placein eşleşip eşleşmediğine baktım
#- sonunda üretim bandı bilgisi var mı ona baktım
#- - var mı diye baktım. Injekto nonun genel formatını kontrol ettim
#
#date
#- injekt nodan üretim tarihi ifade eden kısmı çıkardım
#- kontrol sütununda datenin geçerli bir gün olup olmadığına baktım
#
#time
#- injekt nodan paketlenme saatini çıkardım ve kontrol sütununda kontrol ettim
#- geçerli bir saat değilse yanlış olarak işaretledim
#
#production place check
#- üretilen fabrika ile ürünün eşleşip eşleşmediğine baktım 
#
#durum sütunu
#- herhangi bir kontrol sütunu yanlışsa sorun var hepsi doğruysa sorun yok olarak sınıflandırdım. Daha sonra 
#verileri incelerken bu sütunu kullandım
#- hatalı olanları kırmızıyla işaretleyip düzeltilmiş hali isimli sheette bilgi gerekmeden düzeltilebilecek bilgileri düzelttim (tire, boşluk gibi)
#dışarıdan bilgi gerekenleri de request info sütununda gereken bilgi türü ile belirttim.







# [0.71919024, 0.32587743, 0.92267019] 

env = ReachEnv()

#agent_new = Agent(env)
#replay_buffer = []
eval_log_path = '/home/mert/Desktop/Airlab/airlab_rlbench/RLBench/mertarabaci/reach-ddpg/logs'
eval_callback = EvalCallback(env, best_model_save_path='./logs/dynamicSmall2',
                             log_path='./logs/dynamicSmall2', eval_freq=3000,
                             deterministic=True, render=False)
n_actions = 3
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.005 * np.ones(n_actions))
noise = OrnsteinUhlenbeckActionNoise(mean=np.zeros(n_actions), sigma = 0.005 * np.ones(n_actions)) 
model = DDPG('MlpPolicy', env, action_noise=noise, verbose=1, tensorboard_log=eval_log_path)
#model.learn(total_timesteps=EPISODE_LENGTH*EPISODES, progress_bar=True, callback=eval_callback)
#model.save("dynamicSmall_training")
env.shutdown()

del model # remove to demonstrate saving and loading
env = ReachEnv(headless=False)
model = DDPG.load("/home/mert/Desktop/Airlab/airlab_rlbench/logs/dynamicSmall/best_model.zip")
episodes = 50
for e in range(episodes):
    obs, _ = env.reset()
    for i in range(EPISODE_LENGTH):
        action, _states = model.predict(obs, deterministic=True)
        obs, rewards, dones, terminated, info = env.step(action)
        print(env.euclidean_dist)
        if dones or terminated:
            break



print('Done!')
env.shutdown()
	

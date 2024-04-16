#!/bin/bash
cd /home/chengyiqiu/code/diffusion/Diffuse-Backdoor-Parameters/sh
python ../tools/classfication.py
python ../tools/tg_bot.py --msg 'prepare data done' --title 'prepare'
python ../tools/reload_pdata.py
python ../tools/tg_bot.py --msg 'reload pdata' --title 'reload'
python ../main.py
python ../tools/tg_bot.py --msg 'train diffusion model' --title 'train'

#CUDA_VISIBLE_DEVICES=3 python train.py -s ../waymo_016 --model_path work_dirs/dynamic/nvs/016_seman_bgnn_clip1 --seman_fit_bg 1 --lambda_seman_rgb 1 --start_time 0 --end_time 49 --stride 10
CUDA_VISIBLE_DEVICES=3 python train.py -s ../waymo_016 --model_path work_dirs/dynamic/nvs/016_seman_bgnn_clip2 --seman_fit_bg 1 --lambda_seman_rgb 1 --start_time 50 --end_time 99 --stride 10
CUDA_VISIBLE_DEVICES=3 python train.py -s ../waymo_016 --model_path work_dirs/dynamic/nvs/016_seman_bgnn_clip3 --seman_fit_bg 1 --lambda_seman_rgb 1 --start_time 100 --end_time 149 --stride 10
CUDA_VISIBLE_DEVICES=3 python train.py -s ../waymo_016 --model_path work_dirs/dynamic/nvs/016_seman_bgnn_clip4 --seman_fit_bg 1 --lambda_seman_rgb 1 --start_time 150 --end_time -1 --stride 10

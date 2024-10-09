CUDA_VISIBLE_DEVICES=1 python train.py -s ../waymo_795 --model_path work_dirs/dynamic/nvs/795_dtxdxnn_clip2 --force_reload --detach_x_for_dx 1 --start_time 50 --end_time 99 --stride 10
CUDA_VISIBLE_DEVICES=1 python train.py -s ../waymo_795 --model_path work_dirs/dynamic/nvs/795_dtxdxnn_clip3 --force_reload --detach_x_for_dx 1 --start_time 100 --end_time 149 --stride 10
CUDA_VISIBLE_DEVICES=1 python train.py -s ../waymo_795 --model_path work_dirs/dynamic/nvs/795_dtxdxnn_clip4 --force_reload --detach_x_for_dx 1 --start_time 150 --end_time -1 --stride 10
CUDA_VISIBLE_DEVICES=1 python train.py -s ../waymo_795 --model_path work_dirs/dynamic/nvs/795_dtxdxnn_clip1 --force_reload --detach_x_for_dx 1 --start_time 0 --end_time 49 --stride 10

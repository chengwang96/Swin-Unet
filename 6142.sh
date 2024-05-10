CUDA_VISIBLE_DEVICES=0 python train.py --dataset busi --max_epochs 150 --output_dir './model_out' --input_size 256 --base_lr 0.05 --batch_size 24 --cfg 'configs/swin_tiny_patch4_window7_256_lite.yaml' --dataseed 6142

CUDA_VISIBLE_DEVICES=0 python train.py --dataset glas --max_epochs 150 --output_dir './model_out' --input_size 256 --base_lr 0.05 --batch_size 24 --cfg 'configs/swin_tiny_patch4_window7_256_lite.yaml' --dataseed 6142

CUDA_VISIBLE_DEVICES=0 python train.py --dataset chase --max_epochs 150 --output_dir './model_out' --input_size 960 --base_lr 0.05 --batch_size 4 --cfg 'configs/swin_tiny_patch4_window7_960_lite.yaml' --dataseed 6142
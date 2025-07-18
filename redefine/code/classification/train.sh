## Baseline Models

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode baseline \
#     --log_dir models7/baseline/v1_4 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 21 \
#     --last_layer_dim 17 \
#     --num_layers 1 \
#     --lr 0.000019785 \
#     --weight_decay 0.00022366 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode baseline \
#     --log_dir models7/baseline/v1_5 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 21 \
#     --last_layer_dim 17 \
#     --num_layers 1 \
#     --lr 0.000019785 \
#     --weight_decay 0.00022366 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=0 python train.py \
#     --mode baseline \
#     --log_dir models7/baseline/v1_3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 21 \
#     --last_layer_dim 17 \
#     --num_layers 1 \
#     --lr 0.000019785 \
#     --weight_decay 0.00022366 \
#     --batch_size 32


## Heatmap Models

CUDA_VISIBLE_DEVICES=3 python train.py \
    --mode heatmap_nomc \
    --log_dir models \
    --folds fold1 fold2 fold3 fold4 fold5 \
    --hidden_dim 10 \
    --last_layer_dim 9 \
    --num_layers 1 \
    --lr 0.0000058469 \
    --weight_decay 0.00014189 \
    --batch_size 64

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --mode heatmap \
#     --log_dir models7/heatmaps/v1_5 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 19 \
#     --last_layer_dim 7 \
#     --num_layers 0 \
#     --lr 0.0000047426 \
#     --weight_decay 0.0031192 \
#     --batch_size 64

# CUDA_VISIBLE_DEVICES=3 python train.py \
#     --mode heatmap \
#     --log_dir models7/heatmaps/v1_3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 19 \
#     --last_layer_dim 7 \
#     --num_layers 0 \
#     --lr 0.0000047426 \
#     --weight_decay 0.0031192 \
#     --batch_size 64


## Heatmap Only Models

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode heatmap_only \
#     --log_dir models7/heatmap_only/v1_4 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 4 \
#     --last_layer_dim 4 \
#     --num_layers 1 \
#     --lr 0.000016757 \
#     --weight_decay 0.0029549 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode heatmap_only \
#     --log_dir models7/heatmap_only/v1_5 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 4 \
#     --last_layer_dim 4 \
#     --num_layers 1 \
#     --lr 0.000016757 \
#     --weight_decay 0.0029549 \
#     --batch_size 32

# CUDA_VISIBLE_DEVICES=1 python train.py \
#     --mode heatmap_only \
#     --log_dir models7/heatmap_only/v1_3 \
#     --folds fold1 fold2 fold3 fold4 fold5 \
#     --hidden_dim 4 \
#     --last_layer_dim 4 \
#     --num_layers 1 \
#     --lr 0.000016757 \
#     --weight_decay 0.0029549 \
#     --batch_size 32
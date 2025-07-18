# ### Baseline (just spectra)

# CUDA_VISIBLE_DEVICES=7 python test.py \
#     --mode baseline \
#     --log_dir ./models7/baseline/combined \
#     --folds fold1 fold2 fold3 fold4 fold5

# CUDA_VISIBLE_DEVICES=4 python test.py \
#     --mode baseline \
#     --log_dir ./models6/baseline/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5

# CUDA_VISIBLE_DEVICES=4 python test.py \
#     --mode baseline \
#     --log_dir ./models6/baseline/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5


# ### Baseline with heatmaps

# CUDA_VISIBLE_DEVICES=0 python test.py \
#     --mode heatmap \
#     --log_dir ./models7/heatmaps/combined \
#     --folds fold1 fold2 fold3 fold4 fold5

CUDA_VISIBLE_DEVICES=4 python test.py \
     --mode heatmap_nomc \
     --log_dir ./models \
     --folds fold1 fold2 fold3 fold4 fold5

# CUDA_VISIBLE_DEVICES=4 python test.py \
#     --mode heatmap \
#     --log_dir ./models6/heatmaps/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5

# ### Heatmaps only

# CUDA_VISIBLE_DEVICES=0 python test.py \
#     --mode heatmap_only \
#     --log_dir ./models7/heatmap_only/combined \
#     --folds fold1 fold2 fold3 fold4 fold5

# CUDA_VISIBLE_DEVICES=3 python test.py \
#     --mode heatmap_only \
#     --log_dir ./models6/heatmap_only/v2 \
#     --folds fold1 fold2 fold3 fold4 fold5

# CUDA_VISIBLE_DEVICES=3 python test.py \
#     --mode heatmap_only \
#     --log_dir ./models6/heatmap_only/v3 \
#     --folds fold1 fold2 fold3 fold4 fold5
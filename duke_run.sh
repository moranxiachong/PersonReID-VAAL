
CUDA_VISIBLE_DEVICES=2,3 python tools/train.py \
 --config_file 'configs/duke_se_resnext101_softmax_triplet.yml' \
 INPUT.SIZE_TRAIN '(384, 128)' \
 INPUT.SIZE_TEST '(384, 128)'  \
 DATALOADER.PART -1 \
 OUTPUT_DIR '/raid/home/zhihui/reid_strong_baseline/logs/Duke_se_resnext101_triplet_Adam_init_two_branch_distill_layer4_no_finetune_bz96_unshare_adaptive_lsr_025_viewLoss/'




CUDA_VISIBLE_DEVICES=3 python tools/train.py \
 --config_file 'configs/market_se_resnext101_softmax_triplet.yml' \
 INPUT.SIZE_TRAIN '(384, 128)' \
 INPUT.SIZE_TEST '(384, 128)'  \
 DATALOADER.PART -1 \
 OUTPUT_DIR 'outputs'


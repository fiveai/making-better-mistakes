OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python ../scripts/start_training.py \
    --arch resnet18 \
    --loss ranking-loss \
    --devise True \
    --devise_single_negative False \
    --pretrained False \
    --train_backbone_after 150000 \
    --use_2fc True \
    --fc_inner_dim 512 \
    --use_fc_batchnorm True \
    --weight_decay_fc 5e-4 \
    --lr 1e-6 \
    --lr_fc 1e-4 \
    --data tiered-imagenet-224 \
    --workers 4 \
    --data-paths-config ../data_paths.yml \
    --pretrained_folder crossentropy_tieredimagenet/model_snapshots/checkpoint.epoch0065.pth.tar \
    --output devise_tieredimagenet/ \
    --num_training_steps 200000


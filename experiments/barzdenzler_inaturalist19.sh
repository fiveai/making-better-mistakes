OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python ../scripts/start_training.py \
    --arch resnet18 \
    --loss cosine-plus-xent \
    --barzdenzler True \
    --train_backbone_after 0 \
    --use_2fc False \
    --use_fc_batchnorm True \
    --weight_decay_fc 5e-4 \
    --lr 1e-4 \
    --lr_fc 1e-4 \
    --data inaturalist19-224 \
    --workers 4 \
    --data-paths-config ../data_paths.yml \
    --output barzdenzler_inaturalist19/ \
    --num_training_steps 200000


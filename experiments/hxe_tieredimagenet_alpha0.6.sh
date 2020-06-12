OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 ../scripts/start_training.py \
    --arch resnet18 \
    --loss hierarchical-cross-entropy \
    --lr 1e-5 \
    --alpha 0.6 \
    --data tiered-imagenet-224 \
    --workers 4 \
    --data-paths-config ../data_paths.yml \
    --output hxe_tieredimagenet_alpha0.6/ \
    --num_training_steps 200000


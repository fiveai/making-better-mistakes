OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 python3 ../scripts/start_training.py \
    --arch resnet18 \
    --loss yolo-v2 \
    --lr 1e-5 \
    --data tiered-imagenet-224 \
    --workers 4 \
    --data-paths-config ../data_paths.yml \
    --output yolov2_tieredimagenet/ \
    --num_training_steps 200000


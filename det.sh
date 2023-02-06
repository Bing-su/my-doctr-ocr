python train_det.py \
    --epochs 10 \
    --batch_size 12 \
    --lr 0.001 \
    --wd 0.01 \
    -j 12 \
    --wb \
    --push-to-hub \
    --pretrained \
    --rotation \
    --sched "onecycle" \
    --amp \
    "asdf" "asdf" "db_resnet50"

python train_det.py \
    --epochs 10 \
    --lr 0.001 \
    --wd 0.01 \
    -j 8 \
    --wb \
    --push-to-hub \
    --pretrained \
    --rotation \
    --sched "onecycle" \
    --amp \
    "asdf" "asdf" "db_resnet50"

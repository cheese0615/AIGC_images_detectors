python main.py \
    --train_path 'annotation/datasetv1_train_num2648378.txt' \
    --val_path 'annotation/datasetv1_val_num13309.txt' \
    --lr 0.0001 \
    --num_epochs 30 \
    --weights 'out_dir' \
    --batch_size 128 \
    --data_size 256 \
    --gpu '0,1,2' \
    2>&1 | tee weights/log.log

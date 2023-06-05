python main.py \
    --train_path 'annotation/datasetv1_train_num2648378.txt' \
    --val_path 'annotation/datasetv1_val_num13309.txt' \
    --lr 0.001 \
    --num_epochs 100 \
    --weights 'out_dir' \
    --batch_size 512 \
    --data_size 256 \
    --gpu '1,2' \
    2>&1 | tee weights/log.log

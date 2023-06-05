python main.py \
    --model 'ResNet' \
    --train_file 'annotation/datasetv1_train_num2661687.txt' \
    --num_class 2 \
    --val_ratio 0.005 \
    --test_file 'annotation/datasetv1_test_num295744.txt' \
    --isTrain 0 \
    --lr 0.0001 \
    --resume 'weights/LASTED_pretrained.pt' \
    --data_size 256 \
    --batch_size 48 \
    --gpu '0,1,2' \
    2>&1 | tee weights/log.log

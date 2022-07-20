name=run
CUDA_VISIBLE_DEVICES=0,1,2,3 \
python main.py \
--lr 0.001 \
--checkpoint ${name} \
--batch_size 16 \
--training_epoch 200
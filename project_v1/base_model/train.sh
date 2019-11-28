export BERT_BASE_DIR=./albert_tiny_zh
export TEXT_DIR=./data

python3 run_classifier.py  \
    --task_name=sim  \
    --do_train=true   \
    --do_eval=true   \
    --do_predict=true \
    --data_dir=$TEXT_DIR   \
    --vocab_file=./albert_config/vocab.txt  \
    --bert_config_file=./albert_config/albert_config_tiny.json \
    -max_seq_length=56 \
    --train_batch_size=32   \
    --learning_rate=1e-4  \
    --num_train_epochs=5 \
    --output_dir=model \
    --init_checkpoint=$BERT_BASE_DIR/albert_model.ckpt
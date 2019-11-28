export MODEL_DIR=/home/rcl/PycharmProjects/myproject/albert_test/base_model/model
export BERT_BASE_DIR=/home/rcl/PycharmProjects/myproject/albert_test/base_model/albert_tiny_zh/
python ckpt2pd.py \
    -bert_model_dir $BERT_BASE_DIR \
    -model_dir $MODEL_DIR \
    -max_seq_len 56 \
    -num_labels 4


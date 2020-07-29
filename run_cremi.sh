# CVPPP dataset.
MODEL_ID=$(./assign_model_id.py)
DATASET=cvppp
SAVE_FOLDER=results
BOX_MODEL_ID="box_model_"$DATASET"-"$MODEL_ID
BOX_WEIGHTS=$SAVE_FOLDER/$BOX_MODEL_ID/weights.h5
FULL_MODEL_ID="full_model_"$DATASET"-"$MODEL_ID
FULL_WEIGHTS=$SAVE_FOLDER/$FULL_MODEL_ID/weights.h5
THRESHOLD=30

mkdir -p logs
mkdir -p results

#num_steps 60000, --batch_size 5
#Pretrain attention box controller weights.
./box_model_train.py \
--dataset $DATASET \
--freeze_pretrain_cnn \
--ctrl_cnn_filter_size 3,3,3,3,3,3,3,3 \
--ctrl_cnn_depth 8,8,16,16,32,32,64,64 \
--ctrl_cnn_pool 1,2,1,2,1,2,2,2 \
--num_ctrl_mlp_layers 1 \
--batch_size 2 \
--save_ckpt \
--base_learn_rate 0.001 \
--learn_rate_decay 0.9 \
--steps_per_learn_rate_decay 5000 \
--num_steps 60000 \
--model_id $BOX_MODEL_ID

echo DONE_BOX_MODEL_TRAIN

#Read pretrained weights.
./box_model_read.py \
--model_id $BOX_MODEL_ID \
--results $SAVE_FOLDER \
--output $BOX_WEIGHTS

echo DONE_BOX_MODEL_READ

#num_steps 30000, --batch_size 5
#Train full network.
./full_model_train.py \
--dataset $DATASET \
--use_knob \
--knob_decay 0.5 \
--steps_per_knob_decay 700 \
--knob_box_offset -50000 \
--knob_segm_offset 3000 \
--knob_use_timescale \
--box_loss_fn iou \
--segm_loss_fn iou \
--ctrl_cnn_filter_size 3,3,3,3,3,3,3,3 \
--ctrl_cnn_depth 8,8,16,16,32,32,64,64 \
--ctrl_cnn_pool 1,2,1,2,1,2,2,2 \
--num_ctrl_mlp_layers 1 \
--attn_cnn_filter_size 3,3,3,3,3,3 \
--attn_cnn_depth 8,8,16,16,32,32 \
--attn_cnn_pool 1,2,1,2,1,2 \
--attn_dcnn_filter_size 3,3,3,3,3,3,3 \
--attn_dcnn_depth 32,32,16,16,8,8,1 \
--attn_dcnn_pool 2,1,2,1,2,1,1 \
--filter_height 48 \
--filter_width 48 \
--fixed_gamma \
--stop_canvas_grad \
--batch_size 2 \
--save_ckpt \
--base_learn_rate 0.001 \
--learn_rate_decay 0.8 \
--steps_per_learn_rate_decay 5000 \
--num_steps 30000 \
--ctrl_add_inp \
--ctrl_add_canvas \
--attn_add_inp \
--attn_add_canvas \
--pretrain_ctrl_net $BOX_WEIGHTS \
--model_id $FULL_MODEL_ID

echo DONE_FULL_MODEL_TRAIN

# Run full network evaluation.
./full_model_eval.py \
--model_id $FULL_MODEL_ID \
--dataset $DATASET \
--split 'valid'

echo DONE_FULL_MODEL_EVAL

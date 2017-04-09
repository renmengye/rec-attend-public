# CVPPP dataset.
MODEL_ID=$(./assign_model_id.py)
DATASET=cvppp
SAVE_FOLDER=results
PATCH_MODEL_ID="patch_model_"$DATASET"-"$MODEL_ID
PATCH_WEIGHTS=$SAVE_FOLDER/$PATCH_MODEL_ID/weights.h5
BOX_MODEL_ID="box_model_"$DATASET"-"$MODEL_ID
BOX_WEIGHTS=$SAVE_FOLDER/$BOX_MODEL_ID/weights.h5
FULL_MODEL_ID="full_model_"$DATASET"-"$MODEL_ID
FULL_WEIGHTS=$SAVE_FOLDER/$FULL_MODEL_ID/weights.h5
THRESHOLD=30

mkdir -p logs
mkdir -p results

# Pretrain local window segmentation network weights.
./patch_model_train.py \
--dataset $DATASET \
--attn_box_padding_ratio 0.2 \
--gt_box_ctr_noise 0.05 \
--gt_box_pad_noise 0.1 \
--gt_segm_noise 0.3 \
--attn_cnn_filter_size 3,3,3,3,3,3 \
--attn_cnn_depth 8,8,16,16,32,32 \
--attn_cnn_pool 1,2,1,2,1,2 \
--attn_dcnn_filter_size 3,3,3,3,3,3,3 \
--attn_dcnn_depth 32,32,16,16,8,8,1 \
--attn_dcnn_pool 2,1,2,1,2,1,1 \
--num_attn_mlp_layers 1 \
--filter_height 48 \
--filter_width 48 \
--segm_loss_fn iou \
--save_ckpt \
--base_learn_rate 0.001 \
--batch_size 5 \
--num_steps 2000 \
--attn_cnn_skip 1,0,1,0,1,0 \
--model_id $PATCH_MODEL_ID

# Read pretrained weights.
./patch_model_read.py \
--model_id $PATCH_MODEL_ID \
--output $PATCH_WEIGHTS

# Pretrain attention box controller weights.
./box_model_train.py \
--dataset $DATASET \
--freeze_pretrain_cnn \
--ctrl_cnn_filter_size 3,3,3,3,3,3,3,3 \
--ctrl_cnn_depth 8,8,16,16,32,32,64,64 \
--ctrl_cnn_pool 1,2,1,2,1,2,2,2 \
--num_ctrl_mlp_layers 1 \
--batch_size 5 \
--save_ckpt \
--base_learn_rate 0.001 \
--learn_rate_decay 0.9 \
--steps_per_learn_rate_decay 5000 \
--num_steps 60000 \
--pretrain_cnn $PATCH_WEIGHTS \
--model_id $BOX_MODEL_ID


# Read pretrained weights.
./box_model_read.py \
--model_id $BOX_MODEL_ID \
--results $SAVE_FOLDER \
--output $BOX_WEIGHTS

# Train full network.
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
--batch_size 5 \
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

# Run full network evaluation.
./full_model_eval.py \
--model_id $FULL_MODEL_ID \
--dataset $DATASET \
--split 'valid'
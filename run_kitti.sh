MODEL_ID=$(./assign_model_id.py)
DATASET=kitti
FG_MODEL_ID="fg_model_"$DATASET"-"$MODEL_ID
FG_OUTPUT=results/$FG_MODEL_ID/output
PATCH_MODEL_ID="patch_model_"$DATASET"-"$MODEL_ID
PATCH_WEIGHTS=results/$PATCH_MODEL_ID/weights.h5
BOX_MODEL_ID="box_model_"$DATASET"-"$MODEL_ID
BOX_WEIGHTS=results/$BOX_MODEL_ID/weights.h5
FULL_MODEL_ID="full_model_"$DATASET"-"$MODEL_ID
FULL_WEIGHTS=results/$FULL_MODEL_ID/weights.h5
THRESHOLD_STR='30'
THRESHOLD='0.3'
FG_OUTPUT_THRESH=results/$FG_MODEL_ID/output/$THRESHOLD_STR

# Train FCN preprocessing network.
./fg_model_train.py \
--dataset $DATASET \
--cnn_depth 32,64,64,96,96,128,128,128,128,128,128,128,128,256,256,256,256,512 \
--dcnn_depth 256,256,128,128,96,96,64,64,32,32,9 \
--cnn_skip 1,0,0,0,0,1,0,0,0,0,0,0,0,1,0,0,0,1 \
--dcnn_skip 1,0,1,0,1,0,0,0,0,1 \
--cnn_pool 1,2,1,2,1,2,1,1,1,1,1,1,1,2,1,1,1,2 \
--dcnn_pool 2,1,2,1,2,1,2,1,2,1,1 \
--add_skip_conn \
--segm_loss_fn bce \
--batch_size 8 \
--save_ckpt \
--add_orientation \
--num_steps 40000 \
--optimizer momentum \
--model_id $FG_MODEL_ID

# Pack FCN output prediction.
./fg_pack.py \
--model_id $FG_MODEL_ID \
--dataset $DATASET \
--split 'train,valid,test'

# Render foreground output, used for masking.
./fg_eval.py \
--model_id $FG_MODEL_ID \
--dataset $DATASET \
--split 'valid,test' \
--output $FG_OUTPUT \
--threshold_list $THRESHOLD

# Pretrain attention box controller weights.
./box_model_train.py \
--dataset $DATASET \
--ctrl_cnn_filter_size 3,3,3,3,3,3,3,3 \
--ctrl_cnn_depth 16,16,32,32,64,64,64,64 \
--ctrl_cnn_pool 1,2,1,2,1,2,2,2 \
--num_ctrl_mlp_layers 1 \
--batch_size 5 \
--save_ckpt \
--base_learn_rate 0.001 \
--learn_rate_decay 0.9 \
--steps_per_learn_rate_decay 5000 \
--num_steps 60000 \
--dynamic_var \
--add_d_out \
--add_y_out \
--model_id $BOX_MODEL_ID

# Read pretrained weights.
./box_model_read.py \
--model_id $BOX_MODEL_ID \
--output $BOX_WEIGHTS

# Train full network.
./full_model_train.py \
--dataset $DATASET \
--use_knob \
--knob_decay 0.5 \
--steps_per_knob_decay 1500 \
--knob_box_offset 100 \
--knob_segm_offset 8000 \
--knob_use_timescale \
--box_loss_fn iou \
--segm_loss_fn iou \
--ctrl_cnn_filter_size 3,3,3,3,3,3,3,3 \
--ctrl_cnn_depth 16,16,32,32,64,64,64,64 \
--ctrl_cnn_pool 2,2,1,2,1,2,1,2 \
--num_ctrl_mlp_layers 1 \
--attn_cnn_filter_size 3,3,3,3,3,3 \
--attn_cnn_depth 16,32,32,64,64,96 \
--attn_cnn_pool 1,2,1,2,1,2 \
--attn_dcnn_filter_size 3,3,3,3,3,3,3 \
--attn_dcnn_depth 64,64,32,32,16,16,1 \
--attn_dcnn_pool 2,1,2,1,2,1,1 \
--attn_cnn_skip 1,0,1,0,1,0,1,0 \
--filter_height 48 \
--filter_width 48 \
--save_ckpt \
--num_steps 100000 \
--dynamic_var \
--add_skip_conn \
--add_d_out \
--add_y_out \
--attn_add_d_out \
--attn_add_y_out \
--attn_add_inp \
--attn_add_canvas \
--ctrl_add_d_out \
--ctrl_add_y_out \
--ctrl_add_inp \
--ctrl_add_canvas \
--batch_size 2 \
--base_learn_rate 0.001 \
--learn_rate_decay 0.85 \
--steps_per_learn_rate_decay 5000 \
--stop_canvas_grad \
--pretrain_ctrl_net $BOX_WEIGHTS \
--model_id $FULL_MODEL_ID

# Run full network evaluation.
./full_model_eval.py \
--model_id $FULL_MODEL_ID \
--dataset $DATASET \
--foreground_folder $FG_OUTPUT_THRESH \
--split 'valid,test'

MODEL_ID=$(./assign_model_id.py)
DATASET=cityscapes
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
--cnn_depth 64,96,96,128,128,192,192,256,256,256,256,256,256,256,256,512,512,512,512,512 \
--dcnn_depth 512,512,256,256,192,192,128,128,96,96,64,64,17 \
--cnn_skip 1,0,1,0,1,0,1,0,1,0,0,0,0,0,0,0,0,1,0,0,0 \
--dcnn_skip 1,0,1,0,1,0,1,0,1,0,1,0,0 \
--cnn_pool 1,2,1,2,1,2,1,2,1,1,1,1,1,1,1,2,1,1,1,2 \
--dcnn_pool 2,1,2,1,2,1,2,1,2,1,2,1,1 \
--num_semantic_classes 9 \
--add_skip_conn \
--segm_loss_fn bce \
--batch_size 8 \
--steps_per_valid 100 \
--steps_per_trainval 100 \
--steps_per_plot 200 \
--save_ckpt \
--add_orientation \
--optimizer momentum \
--base_learn_rate 0.01 \
--learn_rate_decay 0.8 \
--steps_per_learn_rate_decay 10000 \
--num_steps 40000 \
--prefetch \
--model_id $FG_MODEL_ID

# Pack FCN output prediction.
./fg_model_pack.py \
--model_id $FG_MODEL_ID \
--dataset $DATASET \
--split 'train,valid,test'

# Pretrain attention box controller weights.
./box_model_train.py --dataset $DATASET \
--ctrl_cnn_filter_size 3,3,3,3,3,3,3,3 \
--ctrl_cnn_depth 16,16,32,32,64,64,64,64 \
--ctrl_cnn_pool 2,2,1,2,1,2,1,2 \
--num_ctrl_mlp_layers 1 \
--save_ckpt \
--dynamic_var \
--add_y_out \
--add_d_out \
--num_semantic_classes 9 \
--batch_size 4 \
--learn_rate_decay 0.85 \
--num_steps 60000 \
--prefetch \
--model_id $BOX_MODEL_ID

# Read pretrained weights.
./box_model_read.py \
--model_id $BOX_MODEL_ID \
--output $BOX_WEIGHTS

./full_model_train.py \
--dataset $DATASET \
--use_knob \
--knob_decay 0.5 \
--knob_box_offset 8000 \
--knob_segm_offset 100 \
--steps_per_knob_decay 1500 \
--knob_use_timescale \
--box_loss_fn iou \
--use_iou_box \
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
--num_ctrl_rnn_iter 5 \
--pretrain_ctrl_net $BOX_WEIGHTS \
--save_ckpt \
--dynamic_var \
--fixed_gamma \
--add_skip_conn \
--add_d_out \
--add_y_out \
--attn_add_d_out \
--attn_add_y_out \
--attn_add_inp \
--attn_add_canvas \
--ctrl_add_d_out \
--ctrl_add_y_out \
--ctrl_add_canvas \
--ctrl_add_inp \
--attn_add_inp \
--attn_add_canvas \
--batch_size 2 \
--learn_rate_decay 0.85 \
--num_semantic_classes 9 \
--stop_canvas_grad \
--base_learn_rate 0.001 \
--num_steps 70000 \
--model_id $FULL_MODEL_ID

# Run evaluation on validation set.
./run_cityscapes_eval.sh "valid" "$FULL_MODEL_ID"

# Run evaluation on test set.
./run_cityscapes_eval.sh "test" "$FULL_MODEL_ID"

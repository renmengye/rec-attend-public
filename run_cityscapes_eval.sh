SPLIT=$1
CSPLIT=""
MODEL_ID=$2
PACK=true
RENDER_GT=""
NOIOU="--no_iou"

if [[ $SPLIT = "valid" ]]; then
  CSPLIT="val"
  RUN_EVAL=true
elif [[ $SPLIT = "test" ]]; then
  CSPLIT="test"
  RUN_EVAL=false
else
  echo "UNKNOWN SPLIT, EXITING."
  exit 1
fi

# Change the cityscapes data path here.
export RESULTS_FOLDER="results"
export CITYSCAPES_RESULTS="$RESULTS_FOLDER"/"$MODEL_ID"/output_"$SPLIT"/cityscapes/
export CITYSCAPES_DATASET="data/cityscapes"
export CITYSCAPES_SPLIT=$CSPLIT

if [[ $CSPLIT ]]; then
  # Pack the network outputs into the HDF5 file.
  #./full_model_pack.py \
  #  --dataset cityscapes \
  #  --split "$SPLIT" \
  #  --model_id "$MODEL_ID" \
  #  --results "$RESULTS_FOLDER" \
  #  --batch_size 4

  if [[ $? -eq 0 ]]; then
    # Run evaluation script, upsample the output to original size.
    ./cityscapes_eval.py \
      --dataset "cityscapes" \
      --split "$SPLIT" \
      --model_id $MODEL_ID \
      --results "$RESULTS_FOLDER" \
      --threshold_list 0.6 \
      --analyzers "" \
      --remove_tiny 1200 \
      --no_iou \
      --lrr_seg \
      --lrr_filename "pretrained/LRR/{}/{}/{}_ss.mat"
    
    if [[ $? -eq 0 ]]; then
      if [[ $RUN_EVAL = true ]]; then
        # Run Cityscapes dataset provided evaluation.
        cd data_api/cityscapes_scripts/evaluation
        python evalInstanceLevelSemanticLabeling.py
        cd ../../../
      fi
    fi
  fi
fi

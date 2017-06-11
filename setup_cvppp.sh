# Set up cityscapes experiments.
# Author: Mengye Ren (mren@cs.toronto.edu)

#########################################################
# Set up folder paths.
# Change lines below.
# Path to CVPPP dataset.
CVPPP_DATA=/ais/gobi4/mren/data/lsc
CVPPP_TEST_DATA=/ais/gobi4/mren/data/lsc_test
# Path to model storage.
SAVE_FOLDER=/ais/gobi5/mren/results/rec-attend
# Path to log storage.
DASHBOARD_LOGS=/u/mren/public_html/results
#########################################################

mkdir -p data

if [ ! \( -e "${data/cvppp}" \) ]
then
  ln -s $CVPPP_DATA data/cvppp
fi

if [ ! \( -e "${data/cvppp_test}" \) ]
then
  ln -s $CVPPP_TEST_DATA data/cvppp_test
fi

FF='results'
if [ ! \( -e "${FF}" \) ]
then
  mkdir -p $SAVE_FOLDER
  ln -s $SAVE_FOLDER results
fi

FF='logs'
if [ ! \( -e "${FF}" \) ]
then
  ln -s $DASHBOARD_LOGS logs
fi

./setup_cvppp.py

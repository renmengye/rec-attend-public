# Change lines below to the path where you store the dataset.
CVPPP_DATA=/ais/gobi4/mren/data/lsc
CVPPP_TEST_DATA=/ais/gobi4/mren/data/lsc_test
mkdir -p data
ln -s $CVPPP_DATA data/cvppp
ln -s $CVPPP_TEST_DATA data/cvppp_test
./setup_cvppp.py

SAVE_FOLDER=/ais/gobi5/mren/results/rec-attend
mkdir -p $SAVE_FOLDER
ln -s $SAVE_FOLDER results

DASHBOARD_LOGS=/u/mren/public_html/results
ln -s $DASHBOARD_LOGS logs
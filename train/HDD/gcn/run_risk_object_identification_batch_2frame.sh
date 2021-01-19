for i in {1..14}
do
    EXPERIMENT_NAME=2021-1-18_083920_w_dataAug_avg
    EPOCH=$i


    CUDA_VISIBLE_DEVICES=0,1 python eval_intervention_test_2frame.py --cause crossing_vehicle \
    --model ./snapshots/crossing_vehicle/$EXPERIMENT_NAME/inputs-camera-epoch-$EPOCH.pth

done
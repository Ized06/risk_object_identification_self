for i in {1..2}
do
    EXPERIMENT_NAME=2020-11-11_184943_w_dataAug_avg
    EPOCH=$i


    CUDA_VISIBLE_DEVICES=0,1 python eval_intervention_test.py --cause crossing_vehicle \
    --model ./snapshots/crossing_vehicle/$EXPERIMENT_NAME/inputs-camera-epoch-$EPOCH.pth

done
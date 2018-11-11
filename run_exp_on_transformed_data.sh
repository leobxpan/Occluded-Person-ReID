python script/experiment/train.py \
-d '(0,)' \
--dataset market1501_transformed \
--normalize_feature false \
-glw 1 \
-llw 0 \
-idlw 0 \
--only_test true \
--exp_dir ~/Experiment/ \
--model_weight_file ~/Dataset/ModelWeights/model_weight.pth 
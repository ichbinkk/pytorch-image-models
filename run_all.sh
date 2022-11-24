#/bin/bash

#Models: efficientnet_b3 efficientnet_b4 pit_xs pit_s swin_vit_t swin_vit_s resnet50 resnet101 resnet152 deit_s deit_b mpvit_tiny mpvit_xsmall vit_t vit_s
# For ablation: dvit_N1 dvit_N2 dvit_N3 dvit_N5

#for model in efficientnet_b3 efficientnet_b4 pit_xs pit_s swin_vit_t swin_vit_s resnet50 resnet152 deit_s deit_b  mpvit_tiny mpvit_xsmall
for model in dvit_N1 dvit_N2 dvit_N3 dvit_N5
do
	python predIE.py --model ${model} --data_dir '../dataset/V4_ec_2' --out_dir './output/V4_ec_2' -e 10 --epochs 10 --ablate True
done

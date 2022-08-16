#/bin/bash

#for model in efficientnet_b3 efficientnet_b4 pit_xs pit_s swin_vit_t swin_vit_s resnet50 resnet101 resnet152 deit_s deit_b mpvit_tiny vit_t vit_s
for model in deit_b mpvit_tiny vit_t vit_s
do
	python predIE.py --model ${model}
done

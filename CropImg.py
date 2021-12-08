from PIL import Image
import os

image_file_path = '../dataset/V3'
out = '../dataset/V3_ec'

# image_file_path = './cam/vit_t'
# model_name = image_file_path.split('/')[-1]
# out = os.path.join('./cam', 'cropped')

if not os.path.exists(out):
    os.makedirs(out)

for root, dirs_name, files_name in os.walk(image_file_path):
        for i in files_name:
            if i.split('.')[-1] in ['png', 'jpg']:
                file_name = os.path.join(root, i) # absolute path for images
                print(file_name)
                img = Image.open(file_name)  # 调用图片
                cropped = img.crop((60, 60, 460, 460))  # (left, upper, right, lower)
                cropped.save(os.path.join(out, i))
                # cropped.save(os.path.join(out, f'{model_name}_{i}'))


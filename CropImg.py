from PIL import Image
import os

image_file_path = '../EC'
out = '../ECC'
for root, dirs_name, files_name in os.walk(image_file_path):
        for i in files_name:
            if i.split('.')[-1]=='png':
                file_name = os.path.join(root, i)
                print(file_name)
                img = Image.open(file_name)  # 调用图片
                cropped = img.crop((0, 0, 600, 700))  # (left, upper, right, lower)
                cropped.save(os.path.join(out, i))

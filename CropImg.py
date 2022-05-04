from PIL import Image
import os


def crop_cam_image(in_dir, out_dir):
    for root, dirs_name, files_name in os.walk(in_dir):
        if files_name:
            dir_name = root.split('/')[-1]
            print(dir_name)
            if dir_name != 'cropped':
                for i in files_name:
                    if i.split('.')[-1] in ['png', 'jpg']:
                        file_name = os.path.join(root, i) # absolute path for images
                        print(file_name)
                        img = Image.open(file_name)  # 调用图片
                        if dir_name == 'efficientnet_b4':
                            cropped = img.crop((0, 0, 310, 300))  # (left, upper, right, lower)
                        else:
                            cropped = img.crop((0, 0, 210, 200))  # (left, upper, right, lower)
                        cropped.save(os.path.join(out_dir, f'{dir_name}_{i}'))


def crop_ec_image(in_dir, out_dir, crop_size):
    for root, dirs_name, files_name in os.walk(in_dir):
        for i in files_name:
            if i.split('.')[-1] in ['png', 'jpg']:
                file_name = os.path.join(root, i) # absolute path for images
                print(file_name)
                img = Image.open(file_name)  # 调用图片
                cropped = img.crop(crop_size)  # (left, upper, right, lower)
                cropped.save(os.path.join(out_dir, i))


if __name__=="__main__":
    # [1] For ec image cropping
    dir_path = '../dataset/png-V4'
    out_path = '../dataset/V4_ec'

    # [2] For Cam image
    # dir_path = './cam/V3'
    # out_path = os.path.join(dir_path, 'cropped')

    os.makedirs(out_path, exist_ok=True)
    if 'cam' in dir_path:
        crop_cam_image(dir_path, out_path)
    elif 'dataset' in dir_path:
        crop_ec_image(dir_path, out_path, (80, 80, 720, 720))


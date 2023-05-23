import os
from PIL import Image

def split_image(image_path, output_folder, rows, columns):
    # 打开原始图像
    image = Image.open(image_path)

    # 获取图像的宽度和高度
    width, height = image.size

    # 计算每个小图像的宽度和高度
    tile_width = width // columns
    tile_height = height // rows

    # 确保输出文件夹存在
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 分割图像并保存小图像
    for y in range(rows):
        for x in range(columns):
            # 计算当前小图像的左上角和右下角坐标
            left = x * tile_width-2
            upper = y * tile_height-4
            right = left + tile_width
            lower = upper + tile_height

            # 提取当前小图像
            tile = image.crop((left, upper, right, lower))

            # 构造保存路径
            save_path = os.path.join(output_folder, f"tile_{x}_{y}.png")

            # 保存小图像
            tile.save(save_path)
            print(f"Saved tile {x}_{y}")

    print("Image splitting complete!")

# 使用示例
image_path = "C:/Users/Nostalgia/Desktop/pkmnbycure.png"  # 输入PNG文件路径
output_folder = "tiles"  # 输出文件夹路径
rows = 17  # 分割为3行
columns = 9  # 分割为4列

split_image(image_path, output_folder, rows, columns)

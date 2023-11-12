from PIL import Image

R = 0
G = 1
B = 2


def process_pixel(pixel):
    r = pixel[R]
    g = pixel[G]
    b = pixel[B]

    return int(g - ((r + b) / 2))


def process_image(image_path, save_path="01_greenness.png"):
    img = Image.open(image_path)
    pixels = list(img.getdata())

    new_pixels = [process_pixel(pixel) for pixel in pixels]

    new_img = Image.new("L", img.size)
    new_img.putdata(new_pixels)
    new_img.save(save_path)

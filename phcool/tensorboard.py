from torch.utils.tensorboard import SummaryWriter
import numpy as np
from PIL import Image

writer = SummaryWriter("logs")
image_path = "dataset\\train\\bees_image\\17209602_fe5a5a746f.jpg"
img_PIL = Image.open(image_path)
img_array = np.array(img_PIL)
# print(img_array.shape)

writer.add_image("test", img_array, 2, dataformats='HWC')

for i in range(100):
    writer.add_scalar("y=2x", 2*i, i)

# writer.add_scalar()
writer.close()

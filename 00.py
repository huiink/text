from PIL import Image
from facenet_pytorch import MTCNN, InceptionResnetV1

# 如果需要，使用 MTCNN 创建人脸检测模型：
mtcnn = MTCNN(image_size=128, margin=0.1)

# 创建一个 inception resnet（在 eval 模式下）：
resnet = InceptionResnetV1(pretrained='vggface2').eval()

img = Image.open("images.jpg")

# 获取裁剪和预白化的图像张量
img_cropped = mtcnn(img, save_path="C:/Users/user/Downloads/")

# 计算嵌入（解压缩以添加批量维度）
img_embedding = resnet(img_cropped.unsqueeze(0))
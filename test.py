import torchvision.transforms as transforms
from PIL import Image
import torchvision.transforms.functional as TF
def get_file(file_name):
    """Load the image by taking file name as input"""
    default_transformation = transforms.Compose([
        transforms.Grayscale(),
        transforms.Resize((64,64))
    ])
    im=Image.open(file_name)
    im.save('static/uploads/inputimage.png')
    image = default_transformation(Image.open(file_name))
    image.save('test/ab/test.png')
    image=Image.open('test/ab/test.png')
    image=TF.to_tensor(image)
    print(image.shape)

get_file("testtemp/100__M_Left_index_finger_CR.BMP")
import numpy
import os                                   # os package
import time                                 # time package
import torch                                # root package
import torch.nn as nn                       # neural networks
import torch.optim as optim                 # Optimizers e.g. gradient descent, ADAM, etc.
import torchvision.datasets as dset         # dataset representation
from torch.utils.data import DataLoader     # dataset loading
import torchvision.transforms as transforms # composable transforms
from torchvision.utils import make_grid, save_image
from torch.autograd import Variable
from PIL import Image
import torchvision.transforms.functional as TF
import torchvision.transforms as T

# %matplotlib inline
import random                               # random package
import numpy as np                          # package for scientific computing

import warnings                             # supress warnings
warnings.filterwarnings('ignore')
#from tqdm import tqdm_notebook as tqdm      # progres package

# random seed everything
def random_seed(seed_value, use_cuda):
    np.random.seed(seed_value) # cpu vars
    torch.manual_seed(seed_value) # cpu vars
    random.seed(seed_value) # Python
    if use_cuda:
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value) # gpu vars
        torch.backends.cudnn.deterministic = True  #needed
        torch.backends.cudnn.benchmark = False

random_seed(17, True)


def weights_init(m):
    """
    Takes as input a neural network m that will initialize all its weights.
    """
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


"""### Generator"""


class Generator(nn.Module):
    def __init__(self, nz=128, channels=3):
        super(Generator, self).__init__()

        self.nz = nz
        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0):
            block = [
                nn.ConvTranspose2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False),
                nn.BatchNorm2d(n_output),
                nn.ReLU(inplace=True),
            ]
            return block

        self.model = nn.Sequential(
            *convlayer(self.nz, 1024, 4, 1, 0),  # Fully connected layer via convolution.
            *convlayer(1024, 512, 4, 2, 1),
            *convlayer(512, 256, 4, 2, 1),
            *convlayer(256, 128, 4, 2, 1),
            *convlayer(128, 64, 4, 2, 1),
            nn.ConvTranspose2d(64, self.channels, 3, 1, 1),
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(-1, self.nz, 1, 1)
        img = self.model(z)
        return img


"""### Discriminator"""


class Discriminator(nn.Module):
    def __init__(self, channels=3):
        super(Discriminator, self).__init__()

        self.channels = channels

        def convlayer(n_input, n_output, k_size=4, stride=2, padding=0, bn=False):
            block = [nn.Conv2d(n_input, n_output, kernel_size=k_size, stride=stride, padding=padding, bias=False)]
            if bn:
                block.append(nn.BatchNorm2d(n_output))
            block.append(nn.LeakyReLU(0.2, inplace=True))
            return block

        self.model = nn.Sequential(
            *convlayer(self.channels, 32, 4, 2, 1),
            *convlayer(32, 64, 4, 2, 1),
            *convlayer(64, 128, 4, 2, 1, bn=True),
            *convlayer(128, 256, 4, 2, 1, bn=True),
            nn.Conv2d(256, 1, 4, 1, 0, bias=False),  # FC with Conv.
        )

    def forward(self, imgs):
        logits = self.model(imgs)
        out = torch.sigmoid(logits)

        return out.view(-1, 1)

criterion = nn.BCELoss()


def create_upload_file(filepath):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Load model weights.
    state_dict = torch.load("model/generator.pth", map_location=device)
    model = Generator()
    model.load_state_dict(state_dict)
    model.eval()
    # Start the verification mode of the model.
    model.eval()

    # Load model weights.
    state_dict = torch.load("model/discriminator.pth")
    modeld = Discriminator()
    modeld.load_state_dict(state_dict)
    modeld.eval()
    # Start the verification mode of the model.
    modeld.eval()
    modeld
    nz = 128
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # load dataset
    #Image_test = file
    batch_size = 1
    image_size = 64

    imagetest = Image.open(filepath)
    imagetest = imagetest.resize((64, 64), Image.ANTIALIAS)
    imagetest = TF.to_tensor(imagetest)
    imagetest = imagetest.unsqueeze_(0)
    netG = model
    netD = modeld
    batch_size = 1
    gen_z = torch.randn(batch_size, nz, 1, 1, device=device)
    gen_images = netG(gen_z)
    images = gen_images.to("cpu").clone().detach()
    # labels = torch.full((batch_size, 1), real_label, device=device)
    print(images.shape)
    print(imagetest.shape)
    realoutput = netD(imagetest)
    fakeoutput = netD(images)
    fakeoutput
    errD = criterion(fakeoutput, realoutput)
    errD.backward()
    realD_x = realoutput.mean().item()
    fakeD_x = realoutput.mean().item()
    print(realoutput)
    print(fakeoutput)
    print(errD)
    imagetest = torch.squeeze(imagetest)
    images = torch.squeeze(images)
    transform = T.ToPILImage()
    imagetest = transform(imagetest)
    fakeimagetest = transform(images)
    imagetest
    fakeimagetest
    realoutput=realoutput.detach().cpu().numpy()
    fakeoutput=fakeoutput.detach().cpu().numpy()
    errD=errD.detach().cpu().numpy()
    print(realoutput)
    realoutput=realoutput[0]
    realoutput = realoutput[0]
    fakeoutput=fakeoutput[0]
    fakeoutput= fakeoutput[0]
    errD=errD
    return imagetest,fakeimagetest,realoutput,fakeoutput,errD,realD_x ,fakeD_x
import cv2
import numpy as np
def convvertimage(filepath):
    img = cv2.imread(filepath)
    a=img.shape
    if a[2] < 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img2 = np.zeros_like(img)
        img2[:,:,0] = gray
        img2[:,:,1] = gray
        img2[:,:,2] = gray
        print(img2.shape)
        cv2.imwrite("test/ab/test.png", img2)
# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    file= 'testtemp/100__M_Left_index_finger_CR.BMP'
    convvertimage(file)
    """real_images,gen_images,realoutput,fakeoutput,errD,realD_x ,fakeD_x=create_upload_file()
    print(real_images)
    print(gen_images)
    print(realoutput)
    print(fakeoutput)

    #file='test/d (378).jpg'
    #create_upload_file(file)

    #file='test/dd (328).jpg'
    #create_upload_file(file)

    #file='test/dsd (405).jpg'
    #create_upload_file(file)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/"""
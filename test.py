import argparse
import time

import cv2
import torchvision
from skimage.metrics import structural_similarity as SSIM
from torch.utils.data import TensorDataset

from Network import *
from dataset import *
from loss import *

parser = argparse.ArgumentParser(description='PyTorch Siamese Reservoir Network')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--reconstruct', type=int, default=0, help='If 1, the denoise images and raw images will be saved at --save_dir')

parser.add_argument('--n_blocks', type=int, default=5)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--save_dir', type=str, default='./output_file/')

args = parser.parse_args()
print(args)

print("Initializing...")
ImageFile.LOAD_TRUNCATED_IMAGES = True

if args.reconstruct == 1:
    reconstruct = True
else:
     reconstruct = False

device = torch.device(args.device)
n_blocks = args.n_blocks
batch_size = args.batch_size

test_cifar = torchvision.datasets.CIFAR10('./data/CIFAR-10/', train=False, download=True)

def noisedataset(test_cifar):
    raw_cifar = np.empty((len(test_cifar), 3, 32, 32))
    noise_cifar = np.empty((len(test_cifar), 3, 32, 32))
    for i, (img,_) in enumerate(test_cifar):
        img = np.array(img)
        raw_cifar[i] = img.transpose(2,0,1) / 255
        noise_cifar[i] = addGaussNoise(img, 30).transpose(2,0,1)
    raw_cifar = torch.from_numpy(raw_cifar)
    noise_cifar = torch.from_numpy(noise_cifar)
    test_set = TensorDataset(raw_cifar, noise_cifar)
    test_iter = DataLoader(test_set, batch_size, num_workers=args.num_workers)
    return test_iter
    
test_iter = noisedataset(test_cifar)


G = Generator(n_blocks)
G.load_state_dict(torch.load('Generator.pth', map_location=device))
G.eval()

D = Discriminator()
D.load_state_dict(torch.load('Discriminator.pth', map_location=device))
D.eval()

G_loss = PerceptualLoss(device)
Regulaztion = RegularizationLoss().to(device)

def calculate_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

print("Initialized")

def test(generator, discriminator, test_iter, loss_g, loss_r, device):
    print("test on", device)
    test_loss = []
    psnr_NC = []
    ssim_NC = []
    test_psnr = []
    test_ssim = []
    generator.to(device)
    discriminator.to(device)
    start = time.time()
    with torch.no_grad():
            for i, (img, nimg) in enumerate(test_iter):
                img, nimg = img.to(device).float(), nimg.to(device).float()
                fakeimg = generator(nimg)
                l_g = loss_g(fakeimg, img, discriminator(fakeimg)) + 2e-8*loss_r(fakeimg)
                test_loss.append(l_g.item())
                psnr_NC.append(calculate_psnr(nimg, img).item())
                test_psnr.append(calculate_psnr(fakeimg, img).item())
                simg = np.uint8(img.detach().cpu().numpy().transpose(0,2,3,1)*255)
                snimg = np.uint8(nimg.detach().cpu().numpy().transpose(0,2,3,1)*255)
                sfimg = np.uint8(fakeimg.detach().cpu().numpy().transpose(0,2,3,1)*255)
                ssim_NC.append(SSIM(simg, snimg, multichannel=True))
                test_ssim.append(SSIM(simg, sfimg, multichannel=True))
            test_avg_loss = np.mean(test_loss)
            psnr_NC_avg = np.mean(psnr_NC)
            test_avg_psnr = np.mean(test_psnr)
            test_avg_ssim = np.mean(test_ssim)
            ssim_NC_avg = np.mean(ssim_NC)
            print(f'Generator Test Loss: {test_avg_loss:.4f}, Cost: {(time.time()-start):.4f}s')
            print(f'PSNR between Raw and Noise img: {psnr_NC_avg:.4f}')
            print(f'SSIM between Raw and Noise img: {ssim_NC_avg:.4f}')
            print(f'PSNR between Raw and Denoised img: {test_avg_psnr:.4f}')
            print(f'SSIM between Raw and Denoised img: {test_avg_ssim:.4f}')
    print("Finished")


def reconstruct(model, test_iter, outPath, device):
    print("Reconstructing")
    model.to(device)
    for i, (img, nimg) in enumerate(test_iter):
        nimg = nimg.to(device).float()
        dnimg = model(nimg)
        img = img.detach().cpu().numpy().transpose(0,2,3,1)
        dnimg = dnimg.detach().cpu().numpy().transpose(0,2,3,1)
        nimg = nimg.detach().cpu().numpy().transpose(0,2,3,1)
        
        for t in range(img.shape[0]):
            rawimgs = Image.fromarray(np.uint8(cv2.normalize(img[t,:,:,:], None, 0, 255, cv2.NORM_MINMAX)))
            nimgs = Image.fromarray(np.uint8(cv2.normalize(nimg[t,:,:,:], None, 0, 255, cv2.NORM_MINMAX)))
            dnimgs = Image.fromarray(np.uint8(cv2.normalize(dnimg[t,:,:,:], None, 0, 255, cv2.NORM_MINMAX)))
            dnimgs.save(outPath + f'{i*batch_size+t}_DN.png')
            nimgs.save(outPath + f'{i*batch_size+t}_N.png')
            rawimgs.save(outPath + f'{i*batch_size+t}.png')
    print("Finished, images are saved at", outPath)
            

if reconstruct:
    test(G, D, test_iter, G_loss, Regulaztion, device)
    reconstruct(G, test_iter, args.save_dir,device)
else:
    test(G, D, test_iter, G_loss, Regulaztion ,device)
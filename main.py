import argparse
import time

import matplotlib.pyplot as plt

from Network import *
from dataset import *
from earlystopping import EarlyStopping
from loss import *

parser = argparse.ArgumentParser(description='SRGAN for Image Denoise')

parser.add_argument('--device', type=str, default='cpu')
parser.add_argument('--trainPath', type=str, default='./data/COCO2014/train2014/')
parser.add_argument('--valPath', type=str, default='./data/COCO2014/val2014/')

parser.add_argument('--n_blocks', type=int, default=5)
parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--lr', type=float, default=1e-3)

args = parser.parse_args()
print(args)

print("Initializing...")
ImageFile.LOAD_TRUNCATED_IMAGES = True

device = torch.device(args.device)
n_blocks = args.n_blocks
n_epochs = args.n_epochs
batch_size = args.batch_size
train_path = args.trainPath
val_path = args.valPath
lr = args.lr

randomcrop = transforms.RandomCrop(96)
train_iter, val_iter = get_data(batch_size, train_path, val_path, randomcrop, 30, ex=1, num_workers=args.num_workers)

G = Generator(n_blocks)
D = Discriminator()

G_loss = PerceptualLoss(device)
Regulaztion = RegularizationLoss().to(device)
D_loss = nn.BCELoss().to(device)

optimizer_g = torch.optim.Adam(G.parameters(), lr=lr*0.1)
optimizer_d = torch.optim.Adam(D.parameters(), lr=lr)

real_label = torch.ones([batch_size, 1, 1, 1]).to(device)
fake_label = torch.zeros([batch_size, 1, 1, 1]).to(device)

early_stopping = EarlyStopping(10, verbose=True)

train_loss_g = []
train_loss_d = []
train_psnr = []
val_loss = []
val_psnr = []

def calculate_psnr(img1, img2):
    return 10. * torch.log10(1. / torch.mean((img1 - img2) ** 2))

print("Initialized")


def train(generator, discriminator, train_iter, val_iter, n_epochs, optim_g, optim_d, loss_g, loss_d, loss_r, device):
    print('train on', device)
    generator.to(device)
    discriminator.to(device)
    cuda = next(generator.parameters()).device
    for epoch in range(n_epochs):
        train_epoch_loss_g = []
        train_epoch_loss_d = []
        train_epoch_psnr = []
        val_epoch_loss = []
        val_epoch_psnr = []
        start = time.time()
        generator.train()
        discriminator.train()
        for i, (img, nimg) in enumerate(train_iter):
            img, nimg = img.to(cuda).float(), nimg.to(cuda).float()
            fakeimg = generator(nimg)

            optim_d.zero_grad()
            realOut = discriminator(img)
            fakeOut = discriminator(fakeimg.detach())
            l_d = loss_d(realOut, real_label) + loss_d(fakeOut, fake_label)
            l_d.backward()
            optim_d.step()

            optim_g.zero_grad()
            l_g = loss_g(fakeimg, img, discriminator(fakeimg)) + 2e-8 * loss_r(fakeimg)
            l_g.backward()
            optim_g.step()

            train_epoch_loss_d.append(l_d.item())
            train_epoch_loss_g.append(l_g.item())
            train_epoch_psnr.append(calculate_psnr(fakeimg, img).item())
        train_epoch_avg_loss_g = np.mean(train_epoch_loss_g)
        train_epoch_avg_loss_d = np.mean(train_epoch_loss_d)
        train_epoch_avg_psnr = np.mean(train_epoch_psnr)
        train_loss_g.append(train_epoch_avg_loss_g)
        train_loss_d.append(train_epoch_avg_loss_d)
        train_psnr.append(train_epoch_avg_psnr)
        print(f'Epoch {epoch + 1}, Generator Train Loss: {train_epoch_avg_loss_g:.4f}, '
              f'Discriminator Train Loss: {train_epoch_avg_loss_d:.4f}, PSNR: {train_epoch_avg_psnr:.4f}')
        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            for i, (img, nimg) in enumerate(val_iter):
                img, nimg = img.to(cuda).float(), nimg.to(cuda).float()
                fakeimg = generator(nimg)
                l_g = loss_g(fakeimg, img, discriminator(fakeimg)) + 2e-8 * loss_r(fakeimg)
                val_epoch_loss.append(l_g.item())
                val_epoch_psnr.append(calculate_psnr(fakeimg, img).item())
            val_epoch_avg_loss = np.mean(val_epoch_loss)
            val_epoch_avg_psnr = np.mean(val_epoch_psnr)
            val_loss.append(val_epoch_avg_loss)
            val_psnr.append(val_epoch_avg_psnr)
            print(
                f'Generator Val Loss: {val_epoch_avg_loss:.4f}, PSNR: {val_epoch_avg_psnr:.4f}, Cost: {(time.time() - start):.4f}s')
            checkpoint_perf = early_stopping(generator, discriminator, train_epoch_avg_psnr, val_epoch_avg_psnr)
            if early_stopping.early_stop:
                print("Early stopping")
                print('Final model performance:')
                print(f'Train PSNR: {checkpoint_perf[0]}, Val PSNR: {checkpoint_perf[1]}')
                break
        torch.cuda.empty_cache()
        
train(G, D, train_iter, val_iter, n_epochs, optimizer_g, optimizer_d, G_loss, D_loss, Regulaztion, device)

plt.figure()
plt.subplot(1,2,1)
plt.plot(train_loss_d, label='Generator Train Loss')
plt.plot(train_loss_g, label='Discriminator Train Loss')
plt.plot(val_loss, label='Validation Loss')
plt.subplot(1,2,2)
plt.plot(train_psnr, label='Train PSNR')
plt.plot(val_psnr, label='Validation PSNR')
plt.title('Training process')
plt.legend()
plt.show()
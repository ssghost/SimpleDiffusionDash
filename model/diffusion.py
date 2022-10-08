import torch
import torchvision
import torch.nn.functional as F
from torchvision import transforms 
from torch.utils.data import DataLoader
from torch import nn
from torch.optim import Adam
from PIL import Image
import math
class Diffusion:
    def __init__(self):
        self.dataset = None
        self.dataloader = None
        self.model = None
        self.img_size = 64
        self.batch = 128
        self.device = "cpu"
        self.post_vari = None
        self.r_alphas = None
        self.c_alphas = None
        self.betas = None

    def load_dataset(self):
        data_transforms = [
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  
            transforms.Lambda(lambda t: (t * 2) - 1) ]
        data_transform = transforms.Compose(data_transforms)

        train = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform)

        test = torchvision.datasets.StanfordCars(root=".", download=True, 
                                         transform=data_transform, split='test')
        self.dataset = torch.utils.data.ConcatDataset([train, test])
        self.dataloader = DataLoader(self.dataset, batch_size=self.batch, shuffle=True, drop_last=True)

    def get_index(self, vals, t, x_shape):
        batch_size = t.shape[0]
        out = vals.gather(-1, t.cpu())
        return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)
    
    def forward_sample(self, t, x_0):
        betas = torch.linspace(0.0001, 0.02, 300)
        self.betas = betas
        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)
        self.r_alphas = torch.sqrt(1. / alphas)
        sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
        sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
        self.c_alphas = sqrt_one_minus_alphas_cumprod 
        self.post_vari = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)
        sqrt_alphas_cumprod_t = self.get_index(sqrt_alphas_cumprod, t, x_0.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index(sqrt_one_minus_alphas_cumprod, t, x_0.shape)
        self.cumprod_alphas = sqrt_one_minus_alphas_cumprod_t
        noise = torch.randn_like(x_0)
        return sqrt_alphas_cumprod_t.to(self.device) * x_0.to(self.device) + sqrt_one_minus_alphas_cumprod_t.to(self.device) * noise.to(self.device), noise.to(self.device)

    def create_model(self):
        class Block(nn.Module):
            def __init__(self, in_ch, out_ch, time_emb_dim, up=False):
                super().__init__()
                self.time_mlp =  nn.Linear(time_emb_dim, out_ch)
                if up:
                    self.conv1 = nn.Conv2d(2*in_ch, out_ch, 3, padding=1)
                    self.transform = nn.ConvTranspose2d(out_ch, out_ch, 4, 2, 1)
                else:
                    self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
                    self.transform = nn.Conv2d(out_ch, out_ch, 4, 2, 1)
                self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
                self.bnorm1 = nn.BatchNorm2d(out_ch)
                self.bnorm2 = nn.BatchNorm2d(out_ch)
                self.relu  = nn.ReLU()
        
            def forward(self, x, t):
                h = self.bnorm1(self.relu(self.conv1(x)))
                time_emb = self.relu(self.time_mlp(t))
                time_emb = time_emb[(..., ) + (None, ) * 2]
                h = h + time_emb
                h = self.bnorm2(self.relu(self.conv2(h)))
                return self.transform(h)


        class SinusoidalPositionEmbeddings(nn.Module):
            def __init__(self, dim):
                super().__init__()
                self.dim = dim

            def forward(self, time):
                device = time.device
                half_dim = self.dim // 2
                embeddings = math.log(10000) / (half_dim - 1)
                embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
                embeddings = time[:, None] * embeddings[None, :]
                embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
                return embeddings


        class SimpleUnet(nn.Module):
            def __init__(self):
                super().__init__()
                image_channels = 3
                down_channels = (64, 128, 256, 512, 1024)
                up_channels = (1024, 512, 256, 128, 64)
                out_dim = 1 
                time_emb_dim = 32

                self.time_mlp = nn.Sequential(
                    SinusoidalPositionEmbeddings(time_emb_dim),
                    nn.Linear(time_emb_dim, time_emb_dim),
                    nn.ReLU()
                )
        
                self.conv0 = nn.Conv2d(image_channels, down_channels[0], 3, padding=1)

                self.downs = nn.ModuleList([Block(down_channels[i], down_channels[i+1], time_emb_dim) for i in range(len(down_channels)-1)])

                self.ups = nn.ModuleList([Block(up_channels[i], up_channels[i+1], time_emb_dim, up=True) for i in range(len(up_channels)-1)])

                self.output = nn.Conv2d(up_channels[-1], 3, out_dim)

            def forward(self, x, timestep):
                t = self.time_mlp(timestep)
                x = self.conv0(x)
                residual_inputs = []
                for down in self.downs:
                    x = down(x, t)
                    residual_inputs.append(x)
                for up in self.ups:
                    residual_x = residual_inputs.pop()
                    x = torch.cat((x, residual_x), dim=1)           
                    x = up(x, t)
                return self.output(x)
        self.model = SimpleUnet()
        print(self.model)

    def get_loss(self, x_0, t):
        x_noisy, noise = self.forward_sample(x_0, t, self.device)
        noise_pred = self.model(x_noisy, t)
        return F.l1_loss(noise, noise_pred)

    @torch.no_grad()
    def timestep_sample(self, x, t):
        betas_t = self.get_index(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = self.get_index(self.c_alphas, t, x.shape)
        sqrt_recip_alphas_t = self.get_index(self.r_alphas, t, x.shape)
        model_mean = sqrt_recip_alphas_t * (x - betas_t * self.model(x, t) / sqrt_one_minus_alphas_cumprod_t)
        posterior_variance_t = self.get_index(self.post_vari, t, x.shape)
    
        if t == 0:
            return model_mean
        else:
            noise = torch.randn_like(x)
            return model_mean + torch.sqrt(posterior_variance_t) * noise   

    def train(self, epochs=200):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model.to(self.device)
        optimizer = Adam(self.model.parameters(), lr=0.001)

        for epoch in range(epochs):
            for step, batch in enumerate(self.dataloader):
                optimizer.zero_grad()
                t = torch.randint(0, 300, (self.batch,), device=self.device).long()
                loss = self.get_loss(self.model, batch[0], t)
                loss.backward()
                optimizer.step()

            if epoch % 5 == 0 and step == 0:
                print(f"Epoch {epoch} | step {step:03d} Loss: {loss.item()} ")

    def test_image(self, ipath = "./input.png", opath = "./output.png"):
        img = Image.open(ipath)
        img_transform = transforms.Compose([
            transforms.Resize((self.img_size, self.img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),  
            transforms.Lambda(lambda t: (t * 2) - 1) ])
        img_t = img_transform(img).float().unsqueeze_(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            out = self.model(img_t)
        out.save(opath)

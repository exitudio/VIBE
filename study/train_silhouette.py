import sys
sys.path.append('../')
from tqdm import tqdm
from pytorch3d.renderer import (TexturesVertex,
                                PointLights,
                                FoVOrthographicCameras,
                                RasterizationSettings,
                                MeshRenderer,
                                MeshRasterizer,
                                SoftSilhouetteShader)
import wandb
import torch.nn as nn
from pytorch3d.structures import Meshes
from lib.models.smpl import get_smpl_faces
import random
from lib.models.spin import projection
from lib.models.spin import hmr, get_pretrained_hmr
from einops import rearrange
import cv2
import glob
from torch.utils.data import Dataset, DataLoader
import numpy as np
import torch


class UnifyLog():
    def __init__(self, config, model):
        wandb.init(project=config['project'],
                   name=config['name'], entity='exitudio', config=config)
        wandb.watch(model)

    def log(self, *args, **kwargs):
        wandb.log(*args, **kwargs)


def train(model, train_loader, optimizer, epoch):
    model.train()
    for i, (norm_imgs, silhouette_imgs, bboxes) in tqdm(enumerate(train_loader)):
        optimizer.zero_grad()
        norm_imgs, silhouette_imgs, bboxes = norm_imgs.to(
            device), silhouette_imgs.to(device), bboxes.to(device)
        norm_imgs = rearrange(norm_imgs, 'b f c h w -> (b f) c h w')
        silhouette_imgs = rearrange(silhouette_imgs, 'b f h w -> (b f) h w')
        bboxes = rearrange(bboxes, 'b f box -> (b f) box')
        output = model(norm_imgs)

        ones = torch.ones(norm_imgs.shape[0], 1).to(device)
        face = torch.tensor(np.expand_dims(faces, axis=0)
                            .astype(np.float32)
                            .repeat(norm_imgs.shape[0], 0)).to(device)
        textures = TexturesVertex(verts_features=torch.ones(
            norm_imgs.shape[0], 6890, 3).to(device))

        transitions = output[0]['theta'][:, 1:3]
        scales = output[0]['theta'][:, 0:1]
        transitions = torch.cat((transitions, ones), dim=-1)
        scales = torch.cat((scales, scales, ones.clone()), dim=-1)

        vert = output[0]['verts']
        mesh = Meshes(vert, face, textures)
        camera = FoVOrthographicCameras(
            device=device, T=transitions, scale_xyz=scales)
        renderer_silhouette = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=camera,
                raster_settings=raster_settings_silhouette
            ),
            shader=SoftSilhouetteShader()
        )
        pred_silhouette_images = renderer_silhouette(
            mesh, cameras=camera, lights=lights)
        pred_silhouette_images = torch.flip(pred_silhouette_images, (1, 2))

        # loss
        # pred_silhouette_images = torch.where(pred_silhouette_images[...,3]>0, 1, 0)
        pred_silhouette_images = pred_silhouette_images[..., 3]
        pred_silhouette_images = torch.ceil(pred_silhouette_images)
        silhouette_imgs = silhouette_imgs/255
        loss = criterion_silhouette(pred_silhouette_images, silhouette_imgs)
        loss.backward()
        optimizer.step()

        unify_log.log({'loss': loss, })


class VideoSilhouetteDataset(Dataset):
    def __init__(self, transform=None):
        PATH = '/home/epinyoan/dataset/casia-b/dataset_b/all/crop_images/*'
        self.folders = list(glob.glob(PATH))
        self.transform = transform

    def __len__(self):
        return len(self.folders)

    def __getitem__(self, idx):
        paths = list(sorted(glob.glob(self.folders[idx]+'/*')))
        start_index = random.randint(0, len(paths)-config['num_frame'])
        norm_imgs = []
        silhouette_imgs = []
        bboxes = []
        for i in range(config['num_frame']):
            data = np.load(paths[start_index+i], allow_pickle=True)
            norm_imgs.append(data['norm_imgs'])
            silhouette_imgs.append(data['silhuette_imgs'])
            bboxes.append(data['bboxes'])
        return np.array(norm_imgs), np.array(silhouette_imgs), np.array(bboxes)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.manual_seed(1)
config = {
    'project': 'Silhouette Training',
    "name": "lr1e-3_ceil",
    'learning_rate': 1e-3,
    'num_frame': 15,
    'batch': 16,
    'epoch': 100
}

vs_dataset = VideoSilhouetteDataset()
train_loader = DataLoader(
    vs_dataset, batch_size=config['batch'], num_workers=1, pin_memory=False, shuffle=True)
model = get_pretrained_hmr()
model = torch.nn.DataParallel(model)
unify_log = UnifyLog(config, model)

# renderer
faces = get_smpl_faces()
lights = PointLights(device=device, location=[[0.0, 0.0, 0.0]])
raster_settings_silhouette = RasterizationSettings(
    image_size=(64, 64),
    blur_radius=1e-6,  # np.log(1. / 1e-4 - 1.)*sigma
    faces_per_pixel=1,
)

# loss
criterion_silhouette = nn.MSELoss().to(device)

optimizer = torch.optim.SGD(
    model.parameters(), lr=config['learning_rate'], momentum=0.9)

for epoch in range(config['epoch']):
    print('epoch:', epoch+1)
    train(model, train_loader, optimizer, epoch)
    # scheduler.step()


# if args.save_model:
#     torch.save(model.state_dict(), "mnist_cnn.pt")

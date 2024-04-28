

# Hae datasetisä hyväksyttävät totuudet

# Hae datasetistä stereokuvat

# Aseta niille joku filteri smoothing tms

# Tee training ja jotain setit

# Treenaa vähä

import torch
from dataset import CityScapesDataset
import random
import math
import torchvision
import time
import tqdm
import numpy
from torchvision.transforms.functional import to_pil_image

torch.cuda.empty_cache()

dataset = CityScapesDataset()

source, truth = dataset[0]

EPOCHES = 200
BATCH_SIZE = 10
DEVICE = 'cuda'

indexes = list(range(len(dataset)))
random.shuffle(indexes)

split_point = math.floor(len(indexes)/20)

train_ds = torch.utils.data.Subset(dataset, indexes[split_point:])
valid_ds = torch.utils.data.Subset(dataset, indexes[0:split_point-1])

loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

vloader = torch.utils.data.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=8)

model = torchvision.models.segmentation.fcn_resnet50(weights=None, num_classes=128)
# model.load_state_dict(torch.load("best_model.pth"), strict=False)

optimizer = torch.optim.AdamW(model.parameters(),
                  lr=1e-4, weight_decay=1e-3)

loss_fn = torch.nn.CrossEntropyLoss()


@torch.no_grad()
def validation(model, loader, loss_fn):
    losses = []
    model.eval()
    for image, target in tqdm.tqdm(loader):
        image, target = image.to(DEVICE), target.float().to(DEVICE)
        output = model(image)['out']
        loss = loss_fn(output, target)
        losses.append(loss.item())

    return numpy.array(losses).mean()

header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'
best_loss = 10000000

for epoch in tqdm.tqdm(range(1, EPOCHES+1)):
    losses = []
    start_time = time.time()
    model.train()
    model.to(DEVICE)
    for image, target in tqdm.tqdm(loader):
        
        image, target = image.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(image)['out']

        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    if epoch % 5 == 0:
        vloss = validation(model, vloader, loss_fn)
        print(header)
        print(raw_line.format(epoch, numpy.array(losses).mean(), vloss,
                                (time.time()-start_time)/60**1))
        losses = []
        
        if vloss < best_loss:
            print("Saved new best model")
            best_loss = vloss
            torch.save(model.state_dict(), 'best_model.pth')
    else:
        print(numpy.array(losses).mean())
    torch.save(model.state_dict(), 'model.pth')
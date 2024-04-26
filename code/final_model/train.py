

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

EPOCHES = 100
BATCH_SIZE = 30
DEVICE = 'cuda'

indexes = list(range(len(dataset)))
random.shuffle(indexes)

split_point = math.floor(len(indexes)/20)

train_ds = torch.utils.data.Subset(dataset, indexes[split_point:])
valid_ds = torch.utils.data.Subset(dataset, indexes[0:split_point-1])

loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=25)

vloader = torch.utils.data.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=25)

model = torchvision.models.segmentation.fcn_resnet50(weights=None, num_classes=1)

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
        # to_pil_image(target[0]).show()
        # to_pil_image(output[0]).show()
        _output = torch.zeros(size=(output.shape[0], output.shape[2], output.shape[3]), dtype=torch.float)
        for i in range(output.shape[0]):
            _output[i, :, :] = output[i, 0, :, :]
        _target = torch.zeros(size=(target.shape[0], target.shape[2], target.shape[3]), dtype=torch.float)
        for i in range(target.shape[0]):
            _target[i, :, :] = target[i, 0, :, :]
        loss = loss_fn(_output, _target)
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
        _output = torch.zeros(size=(output.shape[0], output.shape[2], output.shape[3]), dtype=torch.float)
        for i in range(output.shape[0]):
            _output[i, :, :] = output[i, 0, :, :]
        _target = torch.zeros(size=(target.shape[0], target.shape[2], target.shape[3]), dtype=torch.float)
        for i in range(target.shape[0]):
            _target[i, :, :] = target[i, 0, :, :]

        loss = loss_fn(_output, _target)
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
            torch.save(model.state_dict(), 'model.pth')
    else:
        print(numpy.array(losses).mean())

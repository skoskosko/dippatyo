import torchvision
import numpy
import torch
import argparse
import segmentation_utils
import cv2
from PIL import Image
from dataset import CityScapesDataset, labels
import time
import tqdm

dataset = CityScapesDataset()
torch.cuda.empty_cache()
# torch.cuda.memory.

EPOCHES = 50
BATCH_SIZE = 50
DEVICE = 'cuda'
# torch.cuda.share


# valid_idx, train_idx = [], []
# for i in range(len(dataset)):
#     # if dataset.slices[i][0] == 7:
#     #     valid_idx.append(i)
#     # else:
#     train_idx.append(i)


evens = list(range(0, len(dataset), 2))
odds = list(range(1, len(dataset), 2))

train_ds = torch.utils.data.Subset(dataset, evens)
valid_ds = torch.utils.data.Subset(dataset, odds)

# define training and validation data loaders
loader = torch.utils.data.DataLoader(
    train_ds, batch_size=BATCH_SIZE, shuffle=True, num_workers=20)

vloader = torch.utils.data.DataLoader(
    valid_ds, batch_size=BATCH_SIZE, shuffle=False, num_workers=20)

# print(len(dataset))

# download or load the model from disk
model = torchvision.models.segmentation.fcn_resnet50(pretrained=False, num_classes=31)

# model.load_state_dict(torch.load("model_best.pth"), strict=False)

# model.share_memory()


optimizer = torch.optim.AdamW(model.parameters(),
                  lr=1e-4, weight_decay=1e-3)




class SoftDiceLoss(torch.nn.Module):
    def __init__(self, smooth=1., dims=(-2,-1)):

        super(SoftDiceLoss, self).__init__()
        self.smooth = smooth
        self.dims = dims
    
    def forward(self, x, y):

        tp = (x * y).sum(self.dims)
        fp = (x * (1 - y)).sum(self.dims)
        fn = ((1 - x) * y).sum(self.dims)
        
        dc = (2 * tp + self.smooth) / (2 * tp + fp + fn + self.smooth)
        dc = dc.mean()

        return 1 - dc


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


# bce_fn = torch.nn.BCEWithLogitsLoss()
# dice_fn = SoftDiceLoss()

# def loss_fn(y_pred, y_true):
#     bce = bce_fn(y_pred, y_true)
#     dice = dice_fn(y_pred.sigmoid(), y_true)
#     return 0.8*bce+ 0.2*dice



header = r'''
        Train | Valid
Epoch |  Loss |  Loss | Time, m
'''
#          Epoch         metrics            time
raw_line = '{:6d}' + '\u2502{:7.3f}'*2 + '\u2502{:6.2f}'
best_loss = 10000

for epoch in range(1, EPOCHES+1):
    losses = []
    start_time = time.time()
    model.train()
    model.to(DEVICE)
    for image, target in tqdm.tqdm(loader): # tqdm.tqdm(loader):
        
        image, target = image.to(DEVICE), target.to(DEVICE)

        optimizer.zero_grad()
        output = model(image)['out']
        loss = loss_fn(output, target)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())

    # print(losses)
    if epoch % 5 == 0:
        vloss = validation(model, vloader, loss_fn)
        print(header)
        print(raw_line.format(epoch, numpy.array(losses).mean(), vloss,
                                (time.time()-start_time)/60**1))
        losses = []
        
        if vloss < best_loss:
            best_loss = vloss
            torch.save(model.state_dict(), 'model_best.pth')
    else:
        print(numpy.array(losses).mean())

# # set computation device
# device = 'cuda'

# # model to eval() model and load onto computation devicce
# model.eval().to(device)



# # read the image
# image = Image.open(args['input'])
# # do forward pass and get the output dictionary
# outputs = segmentation_utils.get_segment_labels(image, model, device)
# # get the data from the `out` key
# outputs = outputs['out']
# segmented_image = segmentation_utils.draw_segmentation_map(outputs)
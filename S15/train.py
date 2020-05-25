import torch
from tqdm import tqdm
from utils import show_images
from loss import loss_mask_function, loss_depth_function

def train(model, device, train_loader, optimizer, epoch,
          l1_decay, l2_decay, train_losses, scheduler=None):
  model.train()
  pbar = tqdm(train_loader)
  correct = 0
  processed = 0
  avg_loss = 0
  for batch_idx, sample in enumerate(pbar):
    # get samples
    bg = sample["bg"]
    fg_bg = sample["fg_bg"]
    fg_bg_mask = sample["fg_bg_mask"].to(device)
    fg_bg_depth = sample["fg_bg_depth"].to(device)

    # Init
    optimizer.zero_grad()
    # In PyTorch, we need to set the gradients to zero before starting to do backpropragation because PyTorch accumulates the gradients on subsequent backward passes. 
    # Because of this, when you start your training loop, ideally you should zero out the gradients so that you do the parameter update correctly.

    # Predict
    inp = torch.cat([bg,fg_bg], dim=1)
    inp = inp.to(device)
    mask_pred, depth_pred = model(inp)

    # Calculate loss
    loss_mask, mstr = loss_mask_function(output=mask_pred, target=fg_bg_mask,
                          w_depth=1.0, w_ssim=1.0, w_edge=1.0)
    loss_depth, dstr = loss_depth_function(output=depth_pred, target=fg_bg_depth,
                          w_depth=1.0, w_ssim=1.0, w_edge=1.0)
    loss = loss_mask + loss_depth
    if l1_decay > 0:
      l1_loss = 0
      for param in model.parameters():
        l1_loss += torch.norm(param,1)
      loss += l1_decay * l1_loss
    if l2_decay > 0:
      l2_loss = 0
      for param in model.parameters():
        l2_loss += torch.norm(param,2)
      loss += l2_decay * l2_loss

    # Backpropagation
    loss.backward()
    optimizer.step()
    if scheduler:
      scheduler.step()

    if (batch_idx+1)%500 == 0:
      show_images(mask_pred.detach().cpu(), cols=4, denorm="fg_bg_mask",
        fname=f"images/train_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_Mp.jpg")
      show_images(fg_bg_mask.detach().cpu(), cols=4, denorm="fg_bg_mask",
        fname=f"images/train_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_Mt.jpg")
      show_images(depth_pred.detach().cpu(), cols=4, denorm="fg_bg_depth",
        fname=f"images/train_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_Dp.jpg")
      show_images(fg_bg_depth.detach().cpu(), cols=4, denorm="fg_bg_depth",
        fname=f"images/train_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_Dt.jpg")
      show_images(fg_bg.detach().cpu(), cols=4, denorm="fg_bg",
        fname=f"images/train_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_B.jpg")

    if (batch_idx+1)%500 == 0:
      torch.save(model.state_dict(), f"models/e{epoch:03d}_b{batch_idx:05d}_l{loss.item():0.5f}.pth")

    # Update pbar-tqdm
    # pred = y_pred.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
    # correct += pred.eq(target.view_as(pred)).sum().item()
    processed += len(inp)
    avg_loss += loss.item()

    pbar_str = f'Loss={loss.item():0.5f} LossMask={loss_mask.item():0.5f}[{mstr}] LossDepth={loss_depth.item():05f}[{dstr}] Batch_id={batch_idx}'
    if l1_decay > 0:
      pbar_str = f'L1_loss={l1_loss.item():0.3f} %s' % (pbar_str)
    if l2_decay > 0:
      pbar_str = f'L2_loss={l2_loss.item():0.3f} %s' % (pbar_str)

    pbar.set_description(desc= pbar_str)

  avg_loss /= len(train_loader)
  avg_acc = 100*correct/processed
  train_losses.append(avg_loss)

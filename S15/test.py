import torch
from utils import show_images
from loss import loss_mask_function, loss_depth_function


def test(model, device, test_loader, test_losses, epoch=0, save_img=True):
    model.eval()
    test_loss = 0
    mask_loss = 0
    depth_loss = 0
    eval_m = [0,0,0,0]
    eval_d = [0,0,0,0]
    with torch.no_grad():
        for batch_idx, sample in enumerate(test_loader):
            bg = sample["bg"]
            fg_bg = sample["fg_bg"]
            fg_bg_mask = sample["fg_bg_mask"].to(device)
            fg_bg_depth = sample["fg_bg_depth"].to(device)
            inp = torch.cat([bg,fg_bg], dim=1)
            inp = inp.to(device)
            mask_pred, depth_pred = model(inp)
            # Calculate loss
            loss_mask, mstr = loss_mask_function(output=mask_pred, target=fg_bg_mask,
                                    w_depth=1.0, w_ssim=1.0, w_edge=1.0)
            loss_depth, dstr = loss_depth_function(output=depth_pred, target=fg_bg_depth,
                                    w_depth=1.0, w_ssim=1.0, w_edge=1.0)
            loss = loss_mask + loss_depth
            test_loss += loss.item()
            mask_loss += loss_mask.item()
            depth_loss += loss_depth.item()

            if (batch_idx+1)%500 == 0 and save_img:
                show_images(mask_pred.detach().cpu(), cols=4, denorm="fg_bg_mask",
                    fname=f"images/test_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_Mp.jpg")
                show_images(fg_bg_mask.detach().cpu(), cols=4, denorm="fg_bg_mask",
                    fname=f"images/test_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_Mt.jpg")
                show_images(depth_pred.detach().cpu(), cols=4, denorm="fg_bg_depth",
                    fname=f"images/test_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_Dp.jpg")
                show_images(fg_bg_depth.detach().cpu(), cols=4, denorm="fg_bg_depth",
                    fname=f"images/test_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_Dt.jpg")
                show_images(fg_bg.detach().cpu(), cols=4, denorm="fg_bg",
                    fname=f"images/test_{epoch:03d}_{batch_idx:05d}_{loss.item():0.5f}_B.jpg")

            pem = evaluate(fg_bg_mask, mask_pred)
            ped = evaluate(fg_bg_depth, depth_pred)
            for i in range(len(ped)):
                eval_m[i] += pem[i]
                eval_d[i] += ped[i]

    test_loss /= len(test_loader)
    mask_loss /= len(test_loader)
    depth_loss /= len(test_loader)
    test_losses.append(test_loss)
    eval_t = [0,0,0,0]
    for i in range(len(pem)):
        eval_m[i] /= len(test_loader)
        eval_d[i] /= len(test_loader)
        eval_t[i] = (eval_m[i]+eval_d[i])/2
    
    print('Test set: Average loss: {:.4f}, Average MaskLoss: {:.4f}, Average DepthLoss: {:.4f}\n'.format(
        test_loss, mask_loss, depth_loss))
    print("{}: {:>10}, {:>10}, {:>10}, {:>10}".format(
        "Metric",'t<1.25', 't<1.25^2', 't<1.25^3', 'rms'))
    print("{}: {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(
        "Mask  ",eval_m[0],eval_m[1],eval_m[2],eval_m[3]))
    print("{}: {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}".format(
        "Depth ",eval_d[0],eval_d[1],eval_d[2],eval_d[3]))
    print("{}: {:10.4f}, {:10.4f}, {:10.4f}, {:10.4f}\n".format(
        "Avg   ",eval_t[0],eval_t[1],eval_t[2],eval_t[3]))

def evaluate(gt, pred):
    rmse = torch.sqrt(torch.nn.MSELoss()(gt, pred))
    # abs_rel = torch.mean(torch.abs(gt - pred) / gt)
    
    # While calculating t<1.25, we want the ratio of pixels to within a threshold
    # like 1.25. But if the value of pixel is less than 0.1 then even though the
    # pixel values are close the ratio scale changes
    # For ex, 0.00001 and 0.000001 are very close and we want them to contribute
    # positively for our accuracy but the ratio is 10 which reduces the accuracy.
    # So we clamp the tensors to 0.1 and 1   
    gt = torch.clamp(gt, min=0.1, max=1)
    pred = torch.clamp(pred, min=0.1, max=1)

    thresh = torch.max((gt / pred), (pred / gt))

    a1 = (thresh < 1.25   ).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()

    return a1.item(), a2.item(), a3.item(), rmse.item()


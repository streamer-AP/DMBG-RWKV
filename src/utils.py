import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
import torch.nn as nn
import torch.nn.functional as F
import SimpleITK as sitk
import ttach as tta

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class BoundaryLoss(nn.Module):
    def __init__(self, num_fg_classes, pos_weight=None):
        super().__init__()
        self.num_fg_classes = num_fg_classes
        if pos_weight is not None:
            self.register_buffer('pos_weight', pos_weight)
        else:
            self.pos_weight = None

    @staticmethod
    def _dice_loss_binary(pred, target):
        smooth = 1e-5
        pred_flat = pred.reshape(-1)
        target_flat = target.reshape(-1)
        intersection = (pred_flat * target_flat).sum()
        return 1.0 - (2.0 * intersection + smooth) / (pred_flat.sum() + target_flat.sum() + smooth)

    def forward(self, logit, boundary_gt, label, num_classes):
        total_loss = 0.0
        valid_count = 0

        for c in range(self.num_fg_classes):
            cls_id = c + 1
            class_present = (label == cls_id).any(dim=-1).any(dim=-1)
            if not class_present.any():
                continue

            logit_c = logit[:, c]
            target_c = boundary_gt[:, c]
            mask = class_present.float().unsqueeze(-1).unsqueeze(-1)

            pw = self.pos_weight[c] if self.pos_weight is not None else None
            if pw is not None:
                pw = pw.unsqueeze(0).unsqueeze(0)
            bce = F.binary_cross_entropy_with_logits(
                logit_c, target_c, reduction='none', pos_weight=pw
            )
            bce = (bce * mask).sum() / mask.sum().clamp(min=1.0) / (logit_c.shape[-1] * logit_c.shape[-2])

            pred_c = torch.sigmoid(logit_c)
            dice = self._dice_loss_binary(pred_c * mask, target_c * mask)
            total_loss = total_loss + bce + dice
            valid_count += 1

        if valid_count == 0:
            return logit.sum() * 0.0
        return total_loss / valid_count


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        return dice, hd95
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0
    else:
        return 0, 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, seq_length=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    net_inner = net.module if hasattr(net, 'module') else net
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        d, x, y = image.shape
        net_inner.eval()
        with torch.no_grad():
            if seq_length > 1:
                for start in range(0, d, seq_length):
                    end = min(start + seq_length, d)
                    actual_len = end - start
                    seq_slices = []
                    for ind in range(start, end):
                        slice_img = image[ind, :, :]
                        if x != patch_size[0] or y != patch_size[1]:
                            slice_img = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=3)
                        seq_slices.append(slice_img)
                    while len(seq_slices) < seq_length:
                        seq_slices.append(seq_slices[-1])
                    input_tensor = torch.from_numpy(np.stack(seq_slices)).unsqueeze(1).float().cuda()
                    outputs = net_inner(input_tensor, seq_length=seq_length)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1)
                    for i in range(actual_len):
                        pred_slice = out[i].cpu().detach().numpy()
                        if x != patch_size[0] or y != patch_size[1]:
                            prediction[start + i] = zoom(pred_slice, (x / patch_size[0], y / patch_size[1]), order=0)
                        else:
                            prediction[start + i] = pred_slice
            else:
                for ind in range(d):
                    slice_img = image[ind, :, :]
                    if x != patch_size[0] or y != patch_size[1]:
                        slice_img = zoom(slice_img, (patch_size[0] / x, patch_size[1] / y), order=3)
                    input_tensor = torch.from_numpy(slice_img).unsqueeze(0).unsqueeze(0).float().cuda()
                    outputs = net_inner(input_tensor)
                    if isinstance(outputs, tuple):
                        outputs = outputs[0]
                    out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                    out = out.cpu().detach().numpy()
                    if x != patch_size[0] or y != patch_size[1]:
                        pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                    else:
                        pred = out
                    prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net_inner.eval()
        with torch.no_grad():
            outputs = net_inner(input)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/'+case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/'+ case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/'+ case + "_gt.nii.gz")
    return metric_list

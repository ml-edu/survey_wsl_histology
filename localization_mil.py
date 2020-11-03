import numpy as np
import os
import time
import torch
import torch.nn.functional as F
import cv2
import pickle as pkl

from torch.utils.tensorboard import SummaryWriter
from copy import deepcopy
from torch.optim import SGD, lr_scheduler, Adam
from torch.backends import cudnn
from torchvision.utils import make_grid, save_image
from tqdm import tqdm
from sacred import SETTINGS
from PIL import Image
from torchvision import transforms
from matplotlib import pyplot as plt
from utils import state_dict_to_cpu, AverageMeter
from utils.data.localization.dataset_loaders import dataset_ingredient, load_dataset
from utils.metrics import Evaluator, metric_report
from mil.models import model_ingredient, load_model
from cnn import ResNet_VAE

# Experiment
from sacred import Experiment

ex = Experiment('localization_mil', ingredients=[dataset_ingredient, model_ingredient])

# Filter backspaces and linefeeds
from sacred.utils import apply_backspaces_and_linefeeds

SETTINGS.CAPTURE_MODE = 'sys'
ex.captured_out_filter = apply_backspaces_and_linefeeds

writer = SummaryWriter()

@ex.config
def default_config():
    epochs = 30
    lr = 0.0001
    momentum = 0.9
    weight_decay = 1e-4
    lr_step = 15

    save_dir = os.path.join('results', 'temp')
    dataparallel = True

    # seed = 79187406
    seed = 0

    # balance = 0.000001
    balance = 0.0001

    thresh = 0.2

requires_gradients = ['gradcampp', 'gradcam']

@ex.capture
def get_optimizer_scheduler(parameters, lr, momentum, weight_decay, lr_step):
    # optimizer = SGD(parameters, lr=lr, momentum=momentum, weight_decay=weight_decay,
    #                 nesterov=True if momentum else False)
    optimizer = Adam(parameters, lr=lr)
    scheduler = lr_scheduler.StepLR(optimizer, step_size=lr_step)
    return optimizer, scheduler


def validation(model, loader, device, dataparallel):
    model.eval()
    all_labels = []
    all_probabilities = []
    all_predictions = []
    all_losses = []

    pbar = tqdm(loader, ncols=80, desc='Validation')

    # if ex.current_run.config['model']['pooling'] in requires_gradients:
    #     grad_policy = torch.set_grad_enabled(True)
    # else:
    #     grad_policy = torch.no_grad()

    if dataparallel:
        is_vae = isinstance(model.module.backbone, ResNet_VAE)
    else:
        is_vae = isinstance(model.backbone, ResNet_VAE)

    with torch.no_grad():
        for image, label in pbar:
            image, label = image.to(device), label.to(device)

            if is_vae:
                z, x_reconst, mu, logvar = model.module.backbone(image)
                logits = model.module.pooling(z)
            else:
                logits = model(image)

            if is_vae:
                pred = model.module.pooling.predictions(logits=logits)
                loss = global_loss(image, x_reconst, mu, logvar, logits, label)
                probs = model.module.pooling.probabilities(logits)
            else:
                if dataparallel:
                    pred = model.module.pooling.predictions(logits=logits)
                    loss = model.module.pooling.loss(logits=logits, labels=label)
                    probs = model.module.pooling.probabilities(logits)
                else:
                    pred = model.pooling.predictions(logits=logits)
                    loss = model.pooling.loss(logits=logits, labels=label)
                    probs = model.pooling.probabilities(logits)

            all_labels.append(label.item())
            all_predictions.append(pred.item())
            all_probabilities.append(probs)
            all_losses.append(loss.item())

        all_probabilities = torch.cat(all_probabilities, 0)
        all_probabilities = all_probabilities.detach().cpu()

    metrics = metric_report(np.array(all_labels), all_probabilities.numpy(), np.array(all_predictions))
    metrics['losses'] = np.array(all_losses)

    return metrics


def visualize_cam(mask, img):
    """Make heatmap from mask and synthesize GradCAM result image using heatmap and img.
    Args:
        mask (torch.tensor): mask shape of (1, 1, H, W) and each element has value in range [0, 1]
        img (torch.tensor): img shape of (1, 3, H, W) and each pixel value is in range [0, 1]

    Return:
        heatmap (torch.tensor): heatmap img shape of (3, H, W)
        result (torch.tensor): synthesized GradCAM result of same shape with heatmap.
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask.squeeze()), cv2.COLORMAP_JET)
    heatmap = torch.from_numpy(heatmap).permute(2, 0, 1).float().div(255)
    b, g, r = heatmap.split(1)
    heatmap = torch.cat([r, g, b])

    result = heatmap + img.cpu()
    result = result.div(result.max()).squeeze()

    return heatmap, result

def save_reconst(input, reconst, file_path):
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax1.imshow(input.permute(1, 2, 0))
    ax1.set_title('input')

    ax2.imshow(reconst.permute(1, 2, 0))
    ax2.set_title('reconstruction')

    plt.savefig(file_path)
    plt.close(fig)

def save_visualization(image, mask, saliency_map_0, saliency_map_1, overlay, seg_pred, label, file_path):
    # fig, (ax1, ax2, ax3, ax33, ax4, ax5) = plt.subplots(1, 6, figsize=(20, 10))
    fig, (ax1, ax2, ax4, ax5) = plt.subplots(1, 4, figsize=(20, 10))
    ax1.imshow(image.permute(1, 2, 0))
    ax1.set_title('input')

    ax2.imshow(image.permute(1, 2, 0))
    ax2.imshow(mask, alpha=0.5, cmap='jet')
    ax2.set_title('gt mask')


    # ax3.imshow(saliency_map_0.permute(1, 2, 0))
    # ax3.set_title('img level class saliency map')
    #
    # ax33.imshow(saliency_map_1.permute(1, 2, 0))
    # ax33.set_title('img level class saliency map')

    ax4.imshow(image.permute(1, 2, 0))
    ax4.imshow([saliency_map_0, saliency_map_1][label].permute(1, 2, 0), alpha=0.5, cmap='jet')
    ax4.set_title('CAM overlay')

    ax5.imshow(image.permute(1, 2, 0))
    ax5.imshow(seg_pred, alpha=0.5, cmap='jet')
    ax5.set_title('segmentation prediction')

    plt.savefig(file_path)
    plt.close(fig)


def save_results(save_dir, files, labels, predictions, cams, seg_preds, dice_per_image):
    os.makedirs(save_dir, exist_ok=True)

    with open(save_dir + '/files.pkl', 'wb') as f:
        pkl.dump(files, f)

    with open(save_dir + '/labels.pkl', 'wb') as f:
        pkl.dump(labels, f)

    with open(save_dir + '/preds.pkl', 'wb') as f:
        pkl.dump(predictions, f)

    with open(save_dir + '/cams.pkl', 'wb') as f:
        pkl.dump(cams, f)

    with open(save_dir + '/seg_preds.pkl', 'wb') as f:
        pkl.dump(seg_preds, f)

    with open(save_dir + '/dice_per_image.pkl', 'wb') as f:
        pkl.dump(dice_per_image, f)

@ex.capture
def test(model, loader, device, thresh):
    model.eval()
    all_labels = []
    all_logits = []
    all_predictions = []
    all_losses = []
    all_seg_logits_interp = []
    all_seg_preds_interp = []
    all_dices = []
    all_ious = []
    evaluator = Evaluator(ex.current_run.config['model']['num_classes'])
    image_evaluator = Evaluator(ex.current_run.config['model']['num_classes'])

    pbar = tqdm(loader, ncols=80, desc='Test')

    pooling = ex.current_run.config['model']['pooling']
    if pooling in requires_gradients:
        grad_policy = torch.set_grad_enabled(True)
    else:
        grad_policy = torch.no_grad()

    is_vae = isinstance(model.backbone, ResNet_VAE)

    with grad_policy:
        for i, (image, segmentation, label) in enumerate(pbar):
            image, label = image.to(device), label.to(device)

            if pooling in requires_gradients or pooling == 'ablation':
                model.pooling.eval_cams = True

            if is_vae:
                z, x_reconst, mu, logvar = model.backbone(image)
                logits = model.pooling(z)
            else:
                logits = model(image)

            pred = model.pooling.predictions(logits=logits).item()

            if is_vae:
                loss = global_loss(image, x_reconst, mu, logvar, logits, label)
            else:
                loss = model.pooling.loss(logits=logits, labels=label)

            if ex.current_run.config['dataset']['name'] == 'caltech_birds':
                segmentation_classes = (segmentation.squeeze() > 0.5)
            else:
                segmentation_classes = (segmentation.squeeze() != 0)

            seg_shape = segmentation_classes.shape

            seg_logits = model.pooling.cam.detach().cpu()
            seg_logits_interp = F.interpolate(seg_logits, size=segmentation_classes.shape,
                                              mode='bilinear', align_corners=True).squeeze(0)

            label = label.item()
            all_labels.append(label)
            all_logits.append(logits.cpu())
            all_predictions.append(pred)
            all_losses.append(loss.item())

            if ex.current_run.config['dataset']['name'] == 'glas':
                if ex.current_run.config['model']['pooling'] == 'deepmil_multi':
                    seg_preds_interp = (seg_logits_interp[label] > (1 / seg_logits.numel())).cpu()
                else:
                    # seg_preds_interp = (seg_logits_interp.argmax(0) == label).cpu()
                    seg_preds_interp = (seg_logits_interp[pred] > thresh).cpu()
            else:
                if ex.current_run.config['model']['pooling'] == 'deepmil':
                    seg_preds_interp = (seg_logits_interp.squeeze(0) > (1 / seg_logits.numel())).cpu()
                elif ex.current_run.config['model']['pooling'] == 'deepmil_multi':
                    seg_preds_interp = (seg_logits_interp[label] > (1 / seg_logits.numel())).cpu()
                else:
                    seg_preds_interp = seg_logits_interp.argmax(0).cpu()


            # Save CAMs visualization
            save_dir = 'cams/{}/{}'.format(ex.current_run.config['model']['arch'] +
                                           str(ex.current_run.config['balance']), ex.current_run.config['model']['pooling'])
            os.makedirs(save_dir, exist_ok=True)
            file_path = os.path.join(save_dir, 'cam_{}.png'.format(i))
            seg_logits_interp_norm = seg_logits_interp / seg_logits_interp.max()
            saliency_map_0, overlay_0 = visualize_cam(seg_logits_interp_norm[0], image)
            saliency_map_1, overlay_1 = visualize_cam(seg_logits_interp_norm[1], image)
            overlay = [overlay_0, overlay_1][label]
            save_visualization(image.squeeze().cpu(), segmentation_classes.numpy(), saliency_map_0, saliency_map_1, overlay, seg_preds_interp.numpy() * 255, label,file_path)

            if is_vae:
                x_reconst = x_reconst.detach()
                save_dir = 'reconst/{}/{}'.format(ex.current_run.config['model']['arch'],
                                               ex.current_run.config['model']['pooling'])
                os.makedirs(save_dir, exist_ok=True)
                file_path = os.path.join(save_dir, 'reconst_{}.png'.format(i))
                save_reconst(image.squeeze(0).cpu(), x_reconst.squeeze(0).cpu(), file_path)

            all_seg_logits_interp.append(seg_logits_interp.numpy())
            all_seg_preds_interp.append(seg_preds_interp.numpy().astype('bool'))

            evaluator.add_batch(segmentation_classes, seg_preds_interp)
            image_evaluator.add_batch(segmentation_classes, seg_preds_interp)
            all_dices.append(image_evaluator.dice()[1].item())
            all_ious.append(image_evaluator.intersection_over_union()[1].item())
            image_evaluator.reset()

        if pooling in requires_gradients or pooling == 'ablation':
            model.pooling.eval_cams = False
        all_logits = torch.cat(all_logits, 0)
        all_logits = all_logits.detach()
        all_probabilities = model.pooling.probabilities(all_logits)

    with open('test/gradcampp_seg_preds.pkl', 'wb') as f:
        pkl.dump(all_seg_preds_interp, f)

    results_dir = 'out/{}/{}'.format(ex.current_run.config['model']['arch']
                                     + str(ex.current_run.config['balance']), ex.current_run.config['model']['pooling'])
    save_results(results_dir,
                 loader.dataset.samples,
                 np.array(all_labels),
                 np.array(all_predictions),
                 all_seg_logits_interp,
                 all_seg_preds_interp,
                 np.array(all_dices))

    metrics = metric_report(np.array(all_labels), all_probabilities.numpy(), np.array(all_predictions))
    metrics['images_path'] = loader.dataset.samples
    metrics['labels'] = np.array(all_labels)
    metrics['logits'] = all_logits.numpy()
    metrics['probabilities'] = all_probabilities.numpy()
    metrics['predictions'] = np.array(all_predictions)
    metrics['losses'] = np.array(all_losses)

    metrics['dice_per_image'] = np.array(all_dices)
    metrics['mean_dice'] = metrics['dice_per_image'].mean()
    metrics['dice'] = evaluator.dice()[1].item()
    metrics['iou_per_image'] = np.array(all_ious)
    metrics['mean_iou'] = metrics['iou_per_image'].mean()
    metrics['iou'] = evaluator.intersection_over_union()[1].item()
    metrics['conf_mat'] = evaluator.cm.numpy()

    if ex.current_run.config['dataset']['split'] == 0 and ex.current_run.config['dataset']['fold'] == 0:
        metrics['seg_preds'] = all_seg_preds_interp

    return metrics


@ex.capture
def get_save_name(save_dir, dataset, model):
    exp_name = ex.get_experiment_info()['name']
    start_time = ex.current_run.start_time.strftime('%Y-%m-%d_%H-%M-%S')
    name = '{}_{}_{}_{}_{}'.format(exp_name, ex.current_run._id, dataset['name'], model['pooling'], start_time)
    return os.path.join(save_dir, name)


def vae_loss(x, x_reconst, mu, logvar):
    MSE = F.mse_loss(x_reconst, x, reduction='sum')
    # MSE = F.binary_cross_entropy(x_reconst, x, reduction='sum')
    KLD = -0.5 * torch.mean(1 + logvar - mu**2 - logvar.exp())
    return MSE + KLD

@ex.capture
def global_loss(x, x_reconst, mu, logvar, ypred, y, balance):
    vloss = vae_loss(x, x_reconst, mu, logvar)
    class_loss = F.cross_entropy(ypred, y)
    return balance * vloss + class_loss

@ex.automain
def main(epochs, seed, dataparallel):

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    cudnn.deterministic = True
    torch.manual_seed(seed)

    train_loader, valid_loader, test_loader = load_dataset()
    model = load_model()
    print(model)

    if dataparallel:
        model = torch.nn.DataParallel(model)

    model.to(device)
    optimizer, scheduler = get_optimizer_scheduler(parameters=model.parameters())

    train_losses = AverageMeter()
    train_accs = AverageMeter()

    best_valid_acc = 0
    best_valid_loss = float('inf')
    if dataparallel:
        best_model_dict = deepcopy(model.module.state_dict())
        is_vae = isinstance(model.module.backbone, ResNet_VAE)
    else:
        best_model_dict = deepcopy(model.state_dict())
        is_vae = isinstance(model.backbone, ResNet_VAE)

    for epoch in range(epochs):
        model.train()

        train_losses.reset(), train_accs.reset()
        loader_length = len(train_loader)

        pbar = tqdm(train_loader, ncols=80, desc='Training')
        start = time.time()

        for i, (images, labels) in enumerate(pbar):
            images, labels = images.to(device), labels.to(device, non_blocking=True)

            if is_vae:
                z, x_reconst, mu, logvar = model.module.backbone(images)
                logits = model.module.pooling(z)
            else:
                logits = model(images)


            if is_vae:
                predictions = model.module.pooling.predictions(logits=logits)
                loss = global_loss(images, x_reconst, mu, logvar, logits, labels)
            else:
                if dataparallel:
                    predictions = model.module.pooling.predictions(logits=logits)
                    loss = model.module.pooling.loss(logits=logits, labels=labels)
                else:
                    predictions = model.pooling.predictions(logits=logits)
                    loss = model.pooling.loss(logits=logits, labels=labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            acc = (predictions == labels).float().mean().item()
            loss = loss.item()

            step = epoch + i / loader_length
            ex.log_scalar('training.loss', loss, step)
            ex.log_scalar('training.acc', acc, step)
            train_losses.append(loss)
            train_accs.append(acc)

        scheduler.step()
        end = time.time()
        duration = end - start

        writer.add_scalar('Loss/train', train_losses.avg, epoch)

        # evaluate on validation set
        valid_metrics = validation(model=model, loader=valid_loader, device=device, dataparallel=dataparallel)

        if valid_metrics['losses'].mean() <= best_valid_loss:
            best_valid_acc = valid_metrics['accuracy']
            best_valid_loss = valid_metrics['losses'].mean()
            if dataparallel:
                best_model_dict = deepcopy(model.module.state_dict())
            else:
                best_model_dict = deepcopy(model.state_dict())

        ex.log_scalar('validation.loss', valid_metrics['losses'].mean(), epoch + 1)
        ex.log_scalar('validation.acc', valid_metrics['accuracy'], epoch + 1)

        writer.add_scalar('Loss/val', valid_metrics['losses'].mean(), epoch + 1)

        print('Epoch {:02d} | Duration: {:.1f}s - per batch ({}): {:.3f}s'.format(epoch, duration, loader_length,
                                                                                  duration / loader_length))
        print(' ' * 8, '| Train loss: {:.4f} acc: {:.3f}'.format(train_losses.avg, train_accs.avg))
        print(' ' * 8, '| Valid loss: {:.4f} acc: {:.3f}'.format(valid_metrics['losses'].mean(),
                                                                 valid_metrics['accuracy']))

    # load best model based on validation loss
    model = load_model()
    model.load_state_dict(best_model_dict)
    model.to(device)

    # evaluate on test set
    test_metrics = test(model=model, loader=test_loader, device=device)

    ex.log_scalar('test.loss', test_metrics['losses'].mean(), epochs)
    ex.log_scalar('test.acc', test_metrics['accuracy'], epochs)

    # save model
    # save_name = get_save_name() + '.pickle'
    # torch.save(state_dict_to_cpu(best_model_dict), save_name)
    # ex.add_artifact(os.path.abspath(save_name))

    # save test metrics
    if len(ex.current_run.observers) > 0:
        dataset = ex.current_run.config['dataset']['name']
        pooling = ex.current_run.config['model']['pooling']
        split = ex.current_run.config['dataset']['split']
        fold = ex.current_run.config['dataset']['fold']

        torch.save(
            test_metrics,
            os.path.join(ex.current_run.observers[0].dir, '{}_{}_split-{}_fold-{}.pkl'.format(dataset, pooling,
                                                                                              split, fold))
        )

    # metrics to info.json
    info_to_save = ['labels', 'logits', 'probabilities', 'predictions', 'losses', 'accuracy', 'AP', 'confusion_matrix',
                    'dice', 'dice_per_image', 'mean_dice', 'iou', 'iou_per_image', 'mean_iou', 'conf_mat']
    for k in info_to_save:
        ex.info[k] = test_metrics[k]

    return test_metrics['mean_dice'], test_metrics['dice'], test_metrics['accuracy']

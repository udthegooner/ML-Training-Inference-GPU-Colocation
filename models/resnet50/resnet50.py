# Ignore UserWarnings
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

import argparse, os, random, shutil, time
from enum import Enum

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim
import torch.utils.data
import torch.utils.data.distributed
import torchvision.datasets as datasets
import torchvision.models as models
import torchvision.transforms as transforms
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import Subset
from tqdm import tqdm

# Standardized import for your project structure
from core.performanceIterator import PerformanceIterator

# --- Model Selection ---
model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

# --- Benchmarking Utilities ---
class Summary(Enum):
    NONE = 0
    AVERAGE = 1
    SUM = 2
    COUNT = 3

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f', summary_type=Summary.AVERAGE):
        self.name = name
        self.fmt = fmt
        self.summary_type = summary_type
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

def train(train_loader, model, criterion, optimizer, device, args):
    model.train()
    losses = AverageMeter('Loss', ':.4e')
    start_time = time.time()

    for i, (images, target) in tqdm(enumerate(train_loader), total=args.num_steps):
        if i >= args.num_steps:
            break

        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)

        output = model(images)
        loss = criterion(output, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        losses.update(loss.item(), images.size(0))

    total_time = time.time() - start_time
    print(f'Finished {args.num_steps} steps in {total_time:.2f}s (Avg: {total_time/args.num_steps:.4f}s/it)')

def validate(val_loader, model, criterion, device, args):
    model.eval()
    top1 = AverageMeter('Acc@1', ':6.2f')
    start_time = time.time()

    with torch.no_grad():
        for i, (images, target) in tqdm(enumerate(val_loader), total=args.num_steps):
            if i >= args.num_steps:
                break

            images = images.to(device, non_blocking=True)
            target = target.to(device, non_blocking=True)

            output = model(images)
            acc1, _ = accuracy(output, target, topk=(1, 5))
            top1.update(acc1[0], images.size(0))

    total_time = time.time() - start_time
    print(f'Inference Accuracy: {top1.avg:.2f}%')
    print(f'Finished {args.num_steps} steps in {total_time:.2f}s (Avg: {total_time/args.num_steps:.4f}s/it)')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
    parser.add_argument('--gpuIdx', type=int, default=0, help='GPU id to use')
    parser.add_argument('--alpha', default=0.1, type=float, help='Learning rate (mapped to alpha for consistency)')
    parser.add_argument('--batch_size', default=32, type=int, help='Mini-batch size')
    parser.add_argument('--num_steps', type=int, default=50, help='Number of training steps')
    parser.add_argument('--job_type', type=str, default='training', choices=['training', 'inference'])
    parser.add_argument('--log_file', type=str, default="imagenet.log")
    parser.add_argument('--enable_perf_log', action='store_true', default=True)
    parser.add_argument('--dummy', action='store_true', help="use fake data to benchmark")
    parser.add_argument('--data', metavar='DIR', nargs='?', default='imagenet', help='path to dataset')
    parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50', choices=model_names)
    parser.add_argument('-w', '--workers', default=4, type=int, help='number of data loading workers')
    parser.add_argument('--epochs', default=90, type=int, help='number of total epochs')
    parser.add_argument('--seed', default=42, type=int, help='seed for initializing training')

    args = parser.parse_args()

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.deterministic = True

    DEVICE = torch.device(f'cuda:{args.gpuIdx}' if torch.cuda.is_available() else 'cpu')
    print(f"=> Using device: {DEVICE}")

    # Create model
    print(f"=> creating model '{args.arch}'")
    model = models.__dict__[args.arch]().to(DEVICE)

    # Define loss and optimizer
    criterion = nn.CrossEntropyLoss().to(DEVICE)
    optimizer = torch.optim.SGD(model.parameters(), args.alpha, momentum=0.9, weight_decay=1e-4)
    scheduler = StepLR(optimizer, step_size=30, gamma=0.1)

    # Data loading
    if args.dummy:
        print("=> Using Synthetic (Dummy) Data")
        train_dataset = datasets.FakeData(1281167, (3, 224, 224), 1000, transforms.ToTensor())
        val_dataset = datasets.FakeData(50000, (3, 224, 224), 1000, transforms.ToTensor())
    else:
        traindir = os.path.join(args.data, 'train')
        valdir = os.path.join(args.data, 'val')
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        
        train_dataset = datasets.ImageFolder(traindir, transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ]))
        val_dataset = datasets.ImageFolder(valdir, transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ]))
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.workers, pin_memory=True)

    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)

    # Performance Logging Integration
    if args.enable_perf_log:
        target_loader = val_loader if args.job_type == 'inference' else train_loader
        iterator = PerformanceIterator(target_loader, None, None, None, args.log_file)
    else:
        iterator = val_loader if args.job_type == 'inference' else train_loader
    
    # Run Benchmark
    if args.job_type == 'training':
        train(iterator, model, criterion, optimizer, DEVICE, args)
    else:
        validate(iterator, model, criterion, DEVICE, args)
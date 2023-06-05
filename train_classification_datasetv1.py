import cv2
import albumentations as A
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms, models
from tqdm import tqdm
import argparse
import logging
import PIL.Image as Image
import os
import datetime
import shutil
import numpy as np



# 设置日志格式
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义命令行参数
parser = argparse.ArgumentParser(description='PyTorch Classification Training')
parser.add_argument("--data_root", type=str, default='data/',help="The root folder of training set.")
parser.add_argument('--train_path', type=str, default='train.txt', help='Path to the training annotation file')
parser.add_argument('--val_path', type=str, default='val.txt', help='Path to the validation annotation file')
parser.add_argument('--batch_size', type=int, default=32, help='Batch size for training')
parser.add_argument('--num_epochs', type=int, default=10, help='Number of epochs to train')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum')
parser.add_argument('--weight_decay', type=float, default=1e-4, help='Weight decay')
parser.add_argument('--data_size', type=int, default=256, help='The image size for training.')
parser.add_argument('--log_interval', type=int, default=10, help='Log interval')
parser.add_argument("--weights", type=str, default='out_dir', help="The folder to save models.")
parser.add_argument('--gpu', type=str, default='0,1,2', help='The gpu')
parser.add_argument("--resume", type=str, default='')
args = parser.parse_args()

os.environ['NUMEXPR_MAX_THREADS'] = str(min(os.cpu_count(), os.cpu_count()))
os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

date_now = datetime.datetime.now()
date_now = '/Log_v%02d%02d%02d%02d' % (date_now.month, date_now.day, date_now.hour, date_now.minute)
args.time = date_now
args.out_dir = args.weights+ args.time
if os.path.exists(args.out_dir):
    shutil.rmtree(args.out_dir)
os.makedirs(args.out_dir, exist_ok=True)

logger.info(args)

# 定义数据集类
class ImageDataset(Dataset):
    def __init__(self, data_root, annotation_path, data_size=256):
        self.data_size = data_size
        self.data_root = data_root
        self.annotation_path = annotation_path
        self.data = []
        self.isVal = False
        with open(annotation_path, 'r') as f:
            for line in f:
                path, label = line.strip().split()
                self.data.append((path, int(label)))

        self.albu_pre_train = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.OneOf([
                A.ImageCompression(quality_lower=50, quality_upper=95, compression_type=0, p=1.0),
                # A.GaussianBlur(blur_limit=(3, 7), p=1.0), ######################################### ERROR: Unexpected segmentation fault encountered in worker.
                A.GaussNoise(var_limit=(3.0, 10.0), p=1.0),
                A.ToGray(p=1.0),
            ], p=0.5),
            A.RandomRotate90(p=0.33),
            A.Flip(p=0.33),
        ], p=1.0)
        self.albu_pre_val = A.Compose([
            A.PadIfNeeded(min_height=self.data_size, min_width=self.data_size, p=1.0),
            # A.CenterCrop(height=self.data_size, width=self.data_size, p=1.0),
            A.RandomCrop(height=self.data_size, width=self.data_size, p=1.0), ###########
        ], p=1.0)
        self.imagenet_norm = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((data_size, data_size)),
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])



    def transform(self, x):
        if self.isVal:
            x = self.albu_pre_val(image=x)['image']
        else:
            x = self.albu_pre_train(image=x)['image']
        x = self.imagenet_norm(x)
        return x

    def __len__(self):
        if self.isVal:
            return len(self.data)
        else:
            return len(self.data)

    def __getitem__(self, index):
        if self.isVal:
            return self.getitem(index)
        else:
            return self.getitem(index)

    def getitem(self, index):
        path, label = self.data[index]
        if not os.path.exists(path):
            image_path = os.path.join(self.data_root, path)
        image = cv2.imread(image_path)
        if image is None:
            logger.info('Error Image: %s' % image_path)
            if os.path.exists('error_image.txt'):
                with open('error_image.txt', 'a') as f:
                    f.write(image_path + '\n')
            image = np.zeros([256, 256, 3], dtype=np.uint8)
        image2 = image[..., ::-1]

        crop = self.transform(image2)
        # label = torch.LongTensor([label])
        return crop, label

    def set_val_True(self):
        self.isVal = True

    def set_val_False(self):
        self.isVal = False



# 定义训练函数
def train(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    train_loss = 0
    correct = 0
    total = 0
    train_loader.dataset.set_val_False()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device) ##data [48, 3, 256, 256] target ([48, 1])
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = output.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        if batch_idx % args.log_interval == 0:
            logger.info('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))

    train_loss /= len(train_loader)
    accuracy = 100. * correct / total
    logger.info('Train Epoch: {} Train average loss: {:.4f}, Train accuracy: {}/{} ({:.0f}%)'.format(
        epoch, train_loss, correct, total, accuracy))
    save_path = os.path.join(args.out_dir, f'model_{epoch}.pth')
    torch.save(model.state_dict(), save_path)
    print(f'Saved model to {save_path}')
    return train_loss, accuracy

# 定义验证函数
def validate(model, device, val_loader, criterion):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0
    val_loader.dataset.set_val_True()
    with torch.no_grad():
        for data, target in tqdm(val_loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            val_loss += criterion(output, target).item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

    val_loss /= len(val_loader)
    accuracy = 100. * correct / total
    precision = 100. * correct / predicted.sum().item()
    recall = 100. * correct / target.sum().item()
    f1 = 2 * precision * recall / (precision + recall)
    logger.info('Validation set: Val average loss: {:.4f}, Val accuracy: {}/{} ({:.0f}%), Val precision: {:.4f}, Val recall: {:.4f}, Val f1: {:.4f}'.format(
        val_loss, correct, total, accuracy, precision, recall, f1))
    return val_loss, accuracy, precision, recall, f1

# 主函数
def main():
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info('Device: {}'.format(device))

    # 加载数据集
    train_dataset = ImageDataset(args.data_root, args.train_path)
    val_dataset = ImageDataset(args.data_root, args.val_path)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=20)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=20)

    # 定义模型、损失函数和优化器
    model = models.resnet50(pretrained=True)
    num_features = model.fc.in_features
    model.fc = nn.Linear(num_features, 2)
    model = torch.nn.DataParallel(model).cuda()

    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info('Params: %.2f' % (params / (1024 ** 2)))

    if args.resume != '':
        pretrained = torch.load(args.resume)
        model.load_state_dict(pretrained)

    parameters = [p for p in model.parameters() if p.requires_grad]
    criterion = nn.CrossEntropyLoss().cuda()
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    ##optimizer = optim.Adam(parameters, lr=args.lr)

    # 训练和验证
    for epoch in range(1, args.num_epochs + 1):
        train_loss, train_acc = train(model, device, train_loader, optimizer, criterion, epoch)
        val_loss, val_acc, precision, recall, f1 = validate(model, device, val_loader, criterion)
        print('Epoch: {}, Train Loss: {:.4f}, Train Acc: {:.2f}%, Val Loss: {:.4f}, Val Acc: {:.2f}%, Val Precision: {:.2f}%, Val Recall: {:.2f}%, Val F1: {:.2f}%'.format(
                epoch + 1, train_loss, train_acc, val_loss, val_acc, precision, recall, f1))


if __name__ == '__main__':
    main()
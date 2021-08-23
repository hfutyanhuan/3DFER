import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
# from dataset import Cifar10Dataset
from tensorboardX import SummaryWriter
import torchvision
from data_loader import *
from model import *
from plot import show_acc_curv
from options import Options
from se_resnet import *
from se_resnext import *
from attention_modify import *
import random

def run(device):

    def setup_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        np.random.seed(seed)
        random.seed(seed)
        torch.backends.cudnn.deterministic = True
    # 设置随机数种子
    setup_seed(20)
    args = Options().initialize()
    global_train_acc = []
    global_test_acc = []
    global_train_loss = []
    global_test_loss = []
    # 加载数据集
    #Bosphorus T dataset / facascape subdataset singlechannel
    # train_dataset = NJU3DFE_singlechannel('train')
    # trainloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch_size,
    #     shuffle=args.no_shuffle,
    #     num_workers=4
    # )
    # test_dataset = NJU3DFE_singlechannel('test')
    # testloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=args.test_batch_size,
    #     shuffle=args.no_shuffle,
    #     num_workers=4
    # )
    #Bosphorus dataset
    # train_dataset = Bosphorus_multichannel('train')
    # trainloader = torch.utils.data.DataLoader(
    #     train_dataset,
    #     batch_size=args.train_batch_size,
    #     shuffle=args.no_shuffle,
    #     num_workers=4
    # )
    # test_dataset = Bosphorus_multichannel('test')
    # testloader = torch.utils.data.DataLoader(
    #     test_dataset,
    #     batch_size=args.test_batch_size,
    #     shuffle=args.no_shuffle,
    #     num_workers=4
    # )
    #Facescape subdataset
    train_dataset = NJU3DFE_multichannel('train')
    # print(len(train_dataset ))
    trainloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.train_batch_size,
        shuffle=args.no_shuffle,
        num_workers=4
    )
    test_dataset = NJU3DFE_multichannel('test')
    testloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.test_batch_size,
        shuffle=args.no_shuffle,
        num_workers=4
    )
    net = Resnet18_attention_new(pretrained=not args.no_train).to(device)
    if args.Resume_Model != 'None':
        print('Resume Model: {}'.format(args.Resume_Model))
        checkpoint = torch.load(args.Resume_Model, map_location='cpu')

        net.load_state_dict(checkpoint, strict=False)
    writer = SummaryWriter('./logs_%s' % args.model)
    # for i, data in enumerate(trainloader, 0):
    #     if i==0:
    #         inputs, labels = data
    #         inputs, labels = inputs.to(device), labels.to(device)
    #         writer.add_graph(net,(inputs,))
    
    # print(net)
    criterion = nn.CrossEntropyLoss()

    #params = filter(lambda p: p.requires_grad, net.parameters())
    optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    best_acc = args.best_acc
    print("Start Training, %s!" % args.model)
    with open("acc.txt", "w") as f:
        with open("log.txt", "w")as f2:
            for epoch in range(0, args.Epoch):
                print('\nEpoch: %d' % (epoch + 1))
                net.train()
                sum_loss = 0.0
                correct = 0.0
                total = 0.0
                test_sum_loss = 0.0
                for i, data in enumerate(trainloader, 0):
                    # 准备数据
                    length = len(trainloader)
                    inputs, labels = data
                    inputs, labels = inputs.to(device), labels.to(device)
                    optimizer.zero_grad()
                    # forward + backward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    loss.backward()
                    optimizer.step()
                    sum_loss += loss.item()
                    _, predicted = torch.max(outputs.data, 1)
                    total += labels.size(0)
                    correct += predicted.eq(labels.data).cpu().sum()
                    print('[epoch:%d, iter:%d] Loss: %.03f | Acc: %.3f%% '
                          % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('%03d  %05d |Loss: %.03f | Acc: %.3f%% '
                             % (epoch + 1, (i + 1), sum_loss / (i + 1), 100. * correct / total))
                    f2.write('\n')
                    f2.flush()
                    train_acc = 100. * correct / total
                train_epoch_loss = sum_loss / len(train_dataset)
                train_epoch_acc = correct / len(train_dataset)
                writer.add_scalar('train_loss', train_epoch_loss, epoch + 1)
                writer.add_scalar('train_acc', train_epoch_acc, epoch + 1)
                global_train_acc.append(train_epoch_acc)
                global_train_loss.append(train_epoch_loss)

                print("Waiting Test!")
                with torch.no_grad():
                    correct = 0.0
                    total = 0.0
                    for data in testloader:
                        net.eval()
                        images, labels = data
                        images, labels = images.to(device), labels.to(device)
                        outputs = net(images)
                        loss = criterion(outputs, labels)
                        test_sum_loss += loss.item()
                        # 取得分最高的那个类 (outputs.data的索引号)
                        _, predicted = torch.max(outputs.data, 1)
                        total += labels.size(0)
                        correct += (predicted == labels).sum()
                    
                    print('测试分类准确率为：%.3f%%' % (100. * correct / total))
                    acc = 100. * correct / total
                    test_epoch_loss = test_sum_loss / len(test_dataset)
                    test_epoch_acc = correct / len(test_dataset)
                    writer.add_scalar('test_acc', test_epoch_acc, epoch + 1)
                    writer.add_scalar('test_loss', test_epoch_loss, epoch + 1)
                    # writer.add_scalars('train_test_loss', {'train_loss': train_epoch_loss,'test_loss':test_epoch_loss}, epoch + 1)
                    # writer.add_scalars('train_test_acc', {'train_acc': train_epoch_acc,'test_acc':test_epoch_acc}, epoch + 1)
                    global_test_acc.append(test_epoch_acc)
                    global_test_loss.append(test_epoch_loss)
                    # 将每次测试结果实时写入acc.txt文件中
                    # if acc > max(global_test_acc):
                    print('Saving model......')
                    torch.save(net.state_dict(), '%s/net_%03d_%.3f.pth' % ('./model/', epoch + 1, acc))

                    f.write("EPOCH=%03d,Accuracy= %.3f%%" % (epoch + 1, acc))
                    f.write('\n')
                    f.flush()
                    # 记录最佳测试分类准确率并写入best_acc.txt文件中
                    if acc > best_acc:
                        f3 = open("best_acc.txt", "w")
                        f3.write("EPOCH=%d,best_acc= %.3f%%" % (epoch + 1, acc))
                        f3.close()
                        best_acc = acc
            writer.close()
            # show_acc_curv(length, global_train_acc, global_test_acc)
            # show_acc_curv(length, global_train_loss, global_test_loss)
            print("Training Finished, TotalEPOCH=%d" % args.Epoch)

if __name__ == '__main__':

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    run(device)

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
import numpy as np
import matplotlib.pyplot as plt

#加载测试数据
test_dataset = NJU3DFE_multichannel('test')
testloader = torch.utils.data.DataLoader(
    test_dataset,
    batch_size=10,
    shuffle=False,
    num_workers=0
)
net = Resnet18_attention_new().to('cuda')
net.load_state_dict(torch.load('./model_resnet_lam(78.922%)/net_089.pth'))
with torch.no_grad():
    correct = 0
    total = 0
    for i,data in enumerate(testloader):
        net.eval()
        images, labels = data
        images, labels = images.to('cuda'), labels.to('cuda')
        if i == 0:
            true_labels = labels
        else:
            true_labels  = torch.cat([true_labels,labels],dim=0)
        outputs = net(images)
        # print(outputs.shape)
        # 取得分最高的那个类 (outputs.data的索引号)
        _, predicted = torch.max(outputs.data, 1)
        if i == 0:
            pre_labels = predicted
        else:
            pre_labels  = torch.cat([pre_labels,predicted],dim=0)
        total += labels.size(0)
        correct += (predicted == labels).sum()
    #print(true_labels.shape,pre_labels.shape)
    print('测试分类准确率为：%.3f%%' % (100. * correct / total))

#求解融合矩阵

stacked = torch.stack((true_labels,pre_labels),dim=1)
cmt = torch.zeros(16,16, dtype=torch.float64)
for p in stacked:
    tl, pl = p.tolist()
    cmt[tl, pl] = cmt[tl, pl] + 1
print(cmt/51)
confusion_matrix = cmt/51

confusion_matrix = confusion_matrix.numpy()
#print(confusion_matrix)
#绘制融合矩阵
classes = ['dimpler', 'lip puckerer', 'lip funneler', 'sadness', 'lip roll', 'eye closed', 'brow raiser', 'neutral', 'smile', 
            'mouth stretch', 'anger', 'jaw left', 'jaw right', 'jaw forward', 'mouth left', 'mouth right']
plt.imshow(confusion_matrix, interpolation='nearest', cmap=plt.cm.Oranges)  # 按照像素显示出矩阵 ocean,Oranges
#plt.title('confusion_matrix')
plt.colorbar()

tick_marks = np.arange(len(classes))
plt.xticks(tick_marks, classes, rotation=60)
plt.yticks(tick_marks, classes)
plt.tick_params(labelsize=16)
thresh = confusion_matrix.max() / 2.
# iters = [[i,j] for i in range(len(classes)) for j in range((classes))]
# ij配对，遍历矩阵迭代器

# iters = np.reshape([[[i, j] for j in range(16)] for i in range(16)], (confusion_matrix.size, 2))
# for i, j in iters:
#     plt.text(j, i, format(confusion_matrix[i, j]))  # 显示对应的数字

# plt.ylabel('Real label')
# plt.xlabel('Prediction')
plt.tight_layout()
plt.savefig('confusion_matrix1.png')
plt.show()
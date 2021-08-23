import argparse

class Options():
    def initialize(self):

        parser = argparse.ArgumentParser(description='PyTorch ResNet18 Example')
        #parser.add_argument('--outf', './model/',  help='folder to output images and model checkpoints.')
        #parser.add_argument('--net', './model/Resnet18.pth', help='path to net (to continue training)')
        parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate for SGD,facescape-0.01')
        parser.add_argument('--momentum', type=float, default=0.9, help='SGD momentum')
        parser.add_argument('--train_batch_size', type=int, default=20, metavar='N', help = 'input batch size for training (default: 20),facescape-20')
        parser.add_argument('--test_batch_size', type=int, default=20, metavar='N', help = 'input batch size for testing (default: 20),facescape-20')
        parser.add_argument('--Epoch', type=int, default=50, help='the starting epoch count')
        parser.add_argument('--no_train', action='store_true', default=False, help = 'If train the Model')
        parser.add_argument('--no_shuffle', action='store_true', default=True, help='Whether to disturb the data set')
        parser.add_argument('--save_model', action='store_true', default=False, help = 'For Saving the current Model')
        parser.add_argument('--model', type=str, default='resnet18', help='The model to be trained')
        parser.add_argument('--img_size', type=int, default=112, help='img_size')
        parser.add_argument('--best_acc', type=int, default=20, help='best_acc')
        parser.add_argument('--num_workers', type=int, default=1, help='num_workers')
        parser.add_argument('--dataset', type=str, default='raf', help='dataset-baseline-dataset')
        parser.add_argument('--Resume_Model', type=str, help='Resume_Model', default='E:\yanhuan\AGRA_raw\pretrained-model\preTrainedModel_AGRA_2020\ir18_lfw_112_onlyGlobal.pkl')

        args = parser.parse_args()

        return args

#NJU3DFE  batch_size 20 lr=0.001
#Bosphorus batch_size 5 lr=0.001
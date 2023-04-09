import numpy as np
import torch
import torch.nn as nn
import torch.utils.data as Data
from torchvision import models
from torchvision import transforms
from torchvision import datasets
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix

vgg19 = models.vgg19(pretrained=True)  # 导入预训练好的vgg19模型
vgg = vgg19.features  # 获取vgg16的特征提取层
# 将vgg16的特征提取层参数冻结，不对其进行更新
for param in vgg.parameters():
    param.requires_grad_(False)


# vgg16的特征提取层 + 新的全连接层    组成新的网络
class MyVggModel(nn.Module):
    def __init__(self):
        super(MyVggModel, self).__init__()
        # 预训练的vgg16的特征提取层
        self.vgg = vgg
        # 添加新的全连接层
        self.classifier = nn.Sequential(
            # 第一个全连接层
            nn.Linear(25088, 512, bias=True),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.5),
            # 第二个全连接层
            nn.Linear(512, 256, bias=True),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            # 第三个全连接层
            nn.Linear(256, 19, bias=True),  # 最后有19个类
            # nn.Softmax(dim=1)
        )

    def forward(self, x):
        x = self.vgg(x)
        x = x.view(x.size(0), -1)
        output = self.classifier(x)
        return output


# 准备数据集,对训练集预处理
data_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 裁剪
    # transforms.RandomHorizontalFlip(),  # 翻转
    # transforms.RandomRotation(15),  # 随机旋转-15到15度
    # transforms.RandomGrayscale(),  # 依概率0.1转为灰度图
    transforms.ToTensor()
])
train_data = datasets.ImageFolder('data', transform=data_transform)
train_data_loader = torch.utils.data.DataLoader(train_data, batch_size=16, shuffle=True,
                                                num_workers=0, drop_last=True)
# 对验证集的预处理
test_transform = transforms.Compose([
    transforms.RandomResizedCrop(224),  # 裁剪
    transforms.ToTensor()])
val_data = datasets.ImageFolder('data', transform=test_transform)  # 测试集合
val_data_loader = torch.utils.data.DataLoader(val_data, batch_size=16,
                                              shuffle=True, num_workers=0, drop_last=True)


# 画热力图
def plot_confusion_matrix(cm, labels_name, title):
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]  # 归一化
    plt.imshow(cm, interpolation='nearest')  # 在特定的窗口上显示图像
    plt.title(title)  # 图像标题
    plt.colorbar()
    num_local = np.array(range(len(labels_name)))
    plt.xticks(num_local, labels_name, rotation=90)  # 将标签印在x轴坐标上
    plt.yticks(num_local, labels_name)  # 将标签印在y轴坐标上
    plt.ylabel('True label')
    plt.xlabel('Predicted label')


if __name__ == "__main__":
    classes = (
        'Corals', 'Crabs', 'Dolphin', 'Eel', 'Jelly Fish', 'Lobster', 'Nudibranchs', 'Octopus', 'Penguin', 'Puffers',
        'Sea Rays', 'Sea Urchins', 'Seahorse', 'Seal', 'Sharks', 'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale')

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Myvggc = MyVggModel()
    # Myvggc.load_state_dict(torch.load('params/vgg_pretrained_simple2.pth')) #没有预处理的
    Myvggc.load_state_dict(torch.load('params/vgg_pretrained2.pth'))  # 有预处理的
    Myvggc.to(device)
    # 定义优化器
    optimize = torch.optim.SGD(Myvggc.parameters(), lr=0.001)
    # 定义损失函数
    loss_func = nn.CrossEntropyLoss()

    # 对模型进行训练，所有的数据训练epoch轮
    for epoch in range(1):
        train_loss_epoch = 0
        train_corrects = 0
        Myvggc.train()
        for step, (b_x, b_y) in enumerate(train_data_loader):
            break  # 如果只想评估可以不用训练
            b_x = b_x.to(device)
            b_y = b_y.to(device)
            # 向前传播
            output = Myvggc(b_x).to(device)
            loss = loss_func(output, b_y)
            pre_lab = torch.argmax(output, 1)
            # 向后传播
            optimize.zero_grad()
            loss.backward()
            optimize.step()
            # 计算每个损失和准确率
            train_loss_epoch += loss.item() * b_x.size(0)
            train_corrects += torch.sum(pre_lab == b_y.data)

        # 计算一个epoch的损失和精度
        train_loss = train_loss_epoch / len(train_data.targets)
        print('train loss:epoch:', epoch, train_loss)
        train_acc = train_corrects.double() / len(train_data.targets)
        print('train acc:epoch:', epoch, train_acc)

        # 计算在验证集上的表现
        val_loss_epoch = 0
        val_corrects = 0
        Myvggc.eval()
        # 功能：用于求准确率
        class_correct = list(0. for i in range(19))
        class_total = list(0. for i in range(19))
        for step, (val_x, val_y) in enumerate(val_data_loader):
            val_x = val_x.to(device)
            val_y = val_y.to(device)
            output = Myvggc(val_x).to(device)
            # 计算每个损失和准确率
            loss = loss_func(output, val_y)
            pre_lab = torch.argmax(output, 1)
            val_loss_epoch += loss.item() * val_x.size(0)
            val_corrects += torch.sum(pre_lab == val_y.data)
            # 准确率
            _, predicted = torch.max(output, 1)
            c = (predicted == val_y).squeeze()  # [false true 。。。]行向量
            # f1_score(testset, predicted,average='macro')
            for i in range(16):
                label = val_y[i]
                class_correct[label] += c[i].item()
                class_total[label] += 1

            # 混淆矩阵
            if step == 0:
                true_label = val_y.clone()
                pred_label = pre_lab.clone()
            else:
                true_label = torch.cat((true_label, val_y))
                pred_label = torch.cat((pred_label, pre_lab))
            if step == 150:
                break  # 如果不想画所有数据可以只做150次迭代

        # 准确率
        for i in range(19):
            print('Accuracy of %5s : %2d %%' % (
                classes[i], 100 * class_correct[i] / class_total[i]))
        print('average:', sum(class_correct) / sum(class_total))
        # 计算一个epoch的损失和精度
        val_loss = val_loss_epoch / len(val_data.targets)
        print('val loss:epoch:', epoch, val_loss)
        val_acc = val_corrects.double() / len(val_data.targets)
        print('val acc:epoch:', epoch, val_acc)
        print('\n')
        # 混淆矩阵
        true_label1 = true_label.to('cpu')
        pred_label1 = pred_label.to('cpu')
        cm = confusion_matrix(true_label1, pred_label1)
        # 绘制热力图
        plot_confusion_matrix(cm,
                              ['Corals', 'Crabs', 'Dolphin', 'Eel', 'Jelly Fish', 'Lobster', 'Nudibranchs',
                               'Octopus', 'Penguin', 'Puffers', 'Sea Rays', 'Sea Urchins', 'Seahorse',
                               'Seal', 'Sharks', 'Squid', 'Starfish', 'Turtle_Tortoise', 'Whale'],
                              "Confusion Matrix")
        plt.savefig('img/cm1.png')
        plt.show()
        # 输出混淆矩阵到txt
        f = open('img/predict_matrix.txt', 'w')
        print(str(cm), file=f)
        f.close()
        # F1
        F1 = f1_score(true_label1, pred_label1, average=None)
        print('19个类的F1分数分别是', F1)
        F1 = f1_score(true_label1, pred_label1, average='micro')  # 全局指标
        print('全局的F1分数是', F1)
    # 保存参数
    # torch.save(Myvggc.state_dict(), 'params/vgg_pretrained2.pth')  # 只保存state_dict，不保存完整模型

    print('over')

from dataset import *
import torch.nn as nn
from accuracy import accuracy_check, accuracy_check_for_batch
import os

from model_unet import U_Net #unet
from torch.autograd import Variable


# 训练模型
def train_model(model, data_train, criterion, optimizer):
    """Train the model and report validation error with training error
    Args:
        model: the model to be trained
        criterion: loss function
        data_train (DataLoader): training dataset
    """
    model.train() # 训练前加上，启用 BatchNormalization 和 Dropout
    for batch, (images, masks) in enumerate(data_train):
        #print(images.size()[1])
        images = Variable(images.cuda())
        masks = Variable(masks.cuda())
        outputs = model(images)
        # print(masks.shape, outputs.shape)
        loss = criterion(outputs, masks)
        optimizer.zero_grad() # clear gradients for this training step
        loss.backward() # backpropagation, compute gradients
        optimizer.step() # apply gradients


# 计算训练数据的loss
def get_loss_train(model, data_train, criterion):
    """
        Calculate loss over train set
    """
    model.eval() # 测试时加上，不启用 BatchNormalization 和 Dropout
    total_acc = 0
    total_loss = 0
    for batch, (images, masks) in enumerate(data_train):
        with torch.no_grad(): #推理阶段（预测阶段），不更新梯度
            images = Variable(images.cuda())
            masks = Variable(masks.cuda())
            outputs = model(images)
            loss = criterion(outputs, masks)
            preds = torch.argmax(outputs, dim=1).float() # 选概率高的类别
            acc = accuracy_check_for_batch(masks.cpu(), preds.cpu(), images.size()[0]) # 计算该batch的acc，images.size()[0]=batch_size
            total_acc = total_acc + acc
            total_loss = total_loss + loss.cpu().item()
    return total_acc/(batch+1), total_loss/(batch + 1)


# 预测val
def validate_model(model, data_val, criterion, epoch, make_prediction=True, save_folder_name='prediction'):
    """
        Validation run
    """
    # calculating validation loss
    total_val_loss = 0
    total_val_acc = 0
    for batch, (images_v, masks_v) in enumerate(data_val):
        with torch.no_grad():
            images_v = Variable(images_v.cuda())
            masks_v = Variable(masks_v.cuda())
            #print(images_v.shape, masks_v.shape)
            outputs_v = model(images_v)
            total_val_loss = total_val_loss + criterion(outputs_v, masks_v).cpu().item()
            #print('out', outputs_v.shape)
            outputs_v = torch.argmax(outputs_v, dim=1).float()
            #print(outputs_v.shape)
        if make_prediction:
            im_name = batch  # TODO: Change this to real image name so we know
            pred_msk = save_prediction_image(outputs_v, im_name, epoch, save_folder_name)
            #acc_val = accuracy_check(masks_v, pred_msk)
            acc_val = accuracy_check_for_batch(masks_v.cpu(), outputs_v.cpu(), images_v.size()[0])
            total_val_acc = total_val_acc + acc_val

    return total_val_acc/(batch + 1), total_val_loss/((batch + 1))


def save_prediction_image(outputs_v, im_name, epoch, save_folder_name="result_images"):
    """save images to save_path
    """
    img = outputs_v.cpu().data.numpy() # (256,256,1)
    #print(type(img),img.shape)
    img = img.transpose((1,2,0)) # (1,256,256)
    img = np.squeeze(img, axis=2) # 保留前两维(256,256)
    img = polarize(img)*255
    img_np = img.astype('uint8')
    img = Image.fromarray(img_np, mode='L') #生成灰度图
    #img.show()
    
    # organize images in every epoch
    desired_path = save_folder_name + '/epoch_' + str(epoch) + '/'
    # Create the path if it does not exist
    if not os.path.exists(desired_path):
        os.makedirs(desired_path)
    # Save Image!
    export_name = str(im_name) + '.png'
    img.save(desired_path + export_name)
    
    return img_np


def polarize(img):
    ''' Polarize the value to zero and one
    Args:
        img (numpy): numpy array of image to be polarized
    return:
        img (numpy): numpy array only with zero and one
    '''
    img[img >= 0.5] = 1
    img[img < 0.5] = 0
    return img


if __name__ == "__main__":

    model = U_Net(in_channels=1, out_channels=2)
    model = torch.nn.DataParallel(model, device_ids=list(
    range(torch.cuda.device_count()))).cuda()

    '''
    train = DataTrain(
        ".\\data\\train\\images", ".\\data\\train\\masks") 
    train_load = \
        torch.utils.data.DataLoader(dataset=train,
                                    num_workers=6, batch_size=5, shuffle=True)

    train_model(model, train_load, nn.CrossEntropyLoss(), torch.optim.RMSprop(model.module.parameters(), lr=1e-4))
    '''
    
    val = DataVal(
       './data/val/images', './data/val/masks') 
    val_load = \
        torch.utils.data.DataLoader(dataset=val,
                                    num_workers=3, batch_size=1, shuffle=True)

    val_acc, val_loss = validate_model(
                model, val_load, nn.CrossEntropyLoss(), 1 ,True,"./history") #对val预测
    #print('Val loss:', val_loss, "val acc:", val_acc)
    
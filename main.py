from model_unet import U_Net #unet
from dataset import *
import torch
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
from PIL import Image # 换成opencv
import cv2
from modules import *
from save_history import *

if __name__ == "__main__":
    # Dataset begin
    train = DataTrain(
        './data/train/images', './data/train/masks') # train data
    #test = DataTest(
    #    './data/test/images/') # test data    
    val = DataVal(
       './data/val/images', './data/val/masks') # val data

    # Dataloader begins
    train_load = \
        torch.utils.data.DataLoader(dataset=train,
                                    num_workers=6, batch_size=2, shuffle=True)
    val_load = \
        torch.utils.data.DataLoader(dataset=val,
                                    num_workers=3, batch_size=1, shuffle=False)

    #SEM_test_load = \
    #    torch.utils.data.DataLoader(dataset=SEM_test,
    #                                num_workers=3, batch_size=1, shuffle=False
    # Dataloader end

# Model
    model = U_Net(in_channels=1, out_channels=2)
    #model = U_Net()
    model = torch.nn.DataParallel(model, device_ids=list(
        range(torch.cuda.device_count()))).cuda()

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizerd
    optimizer = torch.optim.RMSprop(model.module.parameters(), lr=1e-4)

    # Parameters
    epoch_start = 0
    epoch_end = 1000

    # Saving History to csv
    header = ['epoch', 'train loss', 'train acc', 'val loss', 'val acc']
    save_file_name = "./history/history.csv"
    save_dir = "./history/"

    # Saving images and models directories
    #model_save_dir = "./history/saved_models"
    image_save_path = "./history/result_images"

    # Train
    print("Initializing Training!")
    for i in range(epoch_start, epoch_end):
        # train the model
        train_model(model, train_load, criterion, optimizer) # 正式训练
        train_acc, train_loss = get_loss_train(model, train_load, criterion) # 计算训练的loss

        #train_loss = train_loss / len(train) 
        print('Epoch', str(i+1), 'Train loss:', train_loss, "Train acc", train_acc) # 输出每一代的loss和acc

        # Validation every 5 epoch 每5代输出一次val集的loss和acc
        if (i+1) % 5 == 0:
            val_acc, val_loss = validate_model(
                model, val_load, criterion, i+1, True, image_save_path) #对val预测
            print('Val loss:', val_loss, "val acc:", val_acc)
            values = [i+1, train_loss, train_acc, val_loss, val_acc]
            export_history(header, values, save_dir, save_file_name) # 将每5代的结果保存
            
            #if (i+1) % 100 == 0:  # save model every 10 epoch 每100代输出一次模型
            #    save_models(model, model_save_dir, i+1)

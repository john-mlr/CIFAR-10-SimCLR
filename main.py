import torch
import torchvision
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

from simclr.modules import SimCLRTransforms
from simclr.modules import SimCLRCIFAR10
from simclr.modules import NT_Xent
from simclr.modules import make_features
from simclr import SimCLR, LinearClassifier

import argparse
import numpy as np
import os

def main():
    parser = argparse.ArgumentParser()
    
    # Distribution parameters
    parser.add_argument('-n', '--nodes', default=1,
                        type=int, metavar='N')
    parser.add_argument('-g', '--gpus', default=1, type=int,
                        help='number of gpus per node')
    parser.add_argument('-nr', '--nr', default=0, type=int,
                        help='ranking within the nodes')
    
    # Model build params
    parser.add_argument('-enc', '--encoder', default='resnet50', type=str,
                        help='Model to use as encoder - can ber either ResNet50 or 18.')
    
    # Training lengths and params
    parser.add_argument('-e', '--epochs', default=50, type=int,
                        help='number of pretraining epochs')
    parser.add_argument('-ee', '-eval_epochs', default=500, type=int,
                        help='number of linear eval epochs')
    parser.add_argument('-ce', '--concurrent_eval', action='store_true',
                        help=('Whether to evaluate the training set throughout pretraining.'
                              'Will slow down pretraining considerably, but provides insight'
                              'into the classification ability of the encoded features'))
    
    # Training hyperparameters
    parser.add_argument('-b', '--batch_size', default=256, type=int,
                        help='total batch size')
    parser.add_argument('-t', '--temperature', default=0.1, type=float,
                        help='NT-Xent loss function temperature.')
    parser.add_argument('-lr', '--learning_rate', default=0.1, type=float,
                        help='Optimizer learning rate')
    
    
    args = parser.parse_args()
    
    args.world_size = args.gpus * args.nodes
    
    # environment variables are set up to run on MSSM's Minvera,
    # running this on local nodes just needs a change to the values
    os.environ['MASTER_PORT'] = '55000'
    os.environ['MASTER_ADDR'] = socket.gethostbyname(socket.gethostname())
    
    mp.spawn(train, nprocs=args.gpus, args=(agrs, ))
    
def train(gpu, args):
    # some distribution setup
    rank = args.nr * agrs.gpus + gpu
    
    dist.init_process_group("nccl", rank=rank, world_size=args.world_size)
    print(f"Process on cuda:{rank} initialized.", flush=True)
    
    torch.manual_seed(201)
    torch.cuda.set_device(gpu)
    
    # adjust batch size for distributed training - will not change if world size is 1
    args.batch_size = int(args.batch_size / args.world_size)
    
    # again, just change path to whatever you would like
    GENERAL_PATH = './'
    
    # dataset setup
    transforms = SimCLRTransforms(img_size=(32,32))
    
    # create two training datasets - one for pretraining and one for 
    # linear evaluation.
    pretrain_dataset = SimCLRCIFAR10(download_path=GENERAL_PATH, train=True, 
                                  transforms=pretraining_transforms,
                                  pretraining=True)
    
    # Linear evaluation (labels, no dual transform)
    train_dataset = SimCLRCIFAR10(download_path=GENERAL_PATH, train=True,
                                  transforms=transforms.eval_transform,
                                  pretraining=False)
    
    test_dataset = SimCLRCIFAR10(download_path=GENERAL_PATH, train=False,
                                 transforms=pretraining_transforms.eval_transform,
                                 pretraining=False)
    
    if dist.get_rank() == 0:
        print("Datasets imported.", flush=True)
        
    # make distributed samplers and dataloaders
    pretrain_sampler = torch.utils.data.distributed.DistributedSampler(pretrain_dataset,
                                                                       num_replicas=args.world_size,
                                                                       rank=rank,
                                                                       shuffle=True,
                                                                       drop_last=True)
    pretrain_loader = torch.utils.data.DataLoader(pretrain_dataset, sampler=pretrain_sampler,
                                                  batch_size=args.batch_size, num_workers=8)
    
    
    # train and test samplers are to pass through the model for feature creation.
    train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset,
                                                                    num_replicas=args.world_size,
                                                                    rank=rank,
                                                                    shuffle=False)
    
    train_loader = torch.utils.data.DataLoader(train_dataset, sampler=train_sampler,
                                               batch_size=args.batch_size, num_workers=4)
    
    test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset,
                                                                   num_replicas=args.world_size,
                                                                   rank=rank,
                                                                   shuffle=False)
    
    test_loader = torch.utils.data.DataLoader(test_dataset, sampler=test_sampler,
                                              batch_size=args.world_size, num_workers=4)
    
    # model setup
    encoder = get_resnet(args)
    num_ftrs = encoder.fc.in_features
    simclr_model = SimCLR(num_ftrs=num_ftrs, encoder=encoder)
    simclr_model = nn.SyncBatchNorm.convert_sync_batchnorm(simclr_model)
    simclr_model = nn.parallel.DistributedDataParallel(simclr_model, device_ids=[gpu])
    
    # Loss function and optimizer setup
    criterion = NT_Xent(batch_size=args.batch_size, temperature=args.temperature).cuda(gpu)
    pretraining_optimizer = torch.optim.Adam(params=simclr_model.parameters(), lr=1e-3)
    
    # Begin Training Loop
    pretraining_losses = []
    for epoch in range(args.epochs):
        if dist.get_rank() == 0:
            print(f"Epoch: {epoch + 1}")
            t_0 = time.time()
            
        epoch_loss = 0
        
        # set the epoch for sample shuffling    
        pretrain_sampler.set_epoch(epoch)
        
        # iterate through all batches and train model
        for batch, (x_i, x_j) in enumerate(pretrain_loader):
            x_i = x_i.cuda()
            x_j = x_j.cuda()
            
            _, _, z_i, z_j = simclr_model(x_i, x_j)
            
            loss = criterion(z_i, z_j)
            
            epoch_loss += loss
            
            loss.backward()
            pretraining_optimizer.step()
            
            #print running statistics every 50 epochs
            if batch % 50 == 0 and dist.get_rank() == 0:
                print("Batch:", batch,
                      "out of", len(pretrain_loader),
                      "Loss:", loss.item(),
                      flush=True)
                
        epoch_loss = epoch_loss.clone().detach()
        dist.reduce(epoch_loss, dst=0)
        
        # print and store the mean epoch loss only on rank 0
        if dist.get_rank() == 0:
            mean_loss = epoch_loss.item() / (dist.get_world_size * len(pretrain_loader))
            
            print("Epoch:", epoch,
                  "Average Loss:", mean_loss)
            
            pretraining_losses.append(mean_loss)
            
    
    # create features from trained SimCLR model
    train_feats, train_labs = make_features(train_loader, simclr_model, gpu, args)
    test_feats, test_labs = make_features(test_loader, simclr_model, gpu, args)
    
    # premake lists on all ranks for distribution
    train_feat_list = [torch.zeros_like(train_feats) for _ in range(args.world_size)]
    train_lab_list = [torch.zeros_like(train_labs) for _ in range(args.world_size)]
    test_feat_list = [torch.zeros_like(test_feats) for _ in range(args.world_size)]
    test_lab_list = [torch.zeros_like(test_labs) for _ in range(args.world_size)]
    
    # gather all features and labels on every device
    dist.all_gather(train_feat_list, train_feats)
    dist.all_gather(train_lab_list, train_labs)
    dist.all_gather(test_feat_list, dev_feats)
    dist.all_gather(test_lab_list, test_labs)
    
    # cat all of the lists (turn them from a list of tensors to a single tensor)
    train_feat_tensor = torch.cat(train_feat_list, dim=0)
    train_lab_tensor = torch.cat(train_lab_list, dim=0)
    test_feat_tensor = torch.cat(test_feat_list, dim=0)
    test_lab_tensor = torch.cat(test_lab_list, dim=0)
    
    # create datasets, samplers and loaders from features
    train_feat_set = torch.utils.data.TensorDataset(train_feat_tensor, train_lab_tensor)
    test_feat_set = torch.utils.data.TensorDataset(test_feat_tensor, test_lab_tensor)
    
    train_feat_sampler = torch.utils.data.distributed.DistributedSampler(train_feat_set, shuffle=True,
                                                                            drop_last=True, num_replicas=args.world_size,
                                                                            rank=rank)
    test_feat_sampler = torch.utils.data.distributed.DistributedSampler(test_feat_set, num_replicas=args.world_size,
                                                                        rank=rank)
    
    train_feat_loader = torch.utils.data.DataLoader(train_feat_set, sampler=train_feat_sampler,
                                                    batch_size=args.batch_size)
    test_feat_loader = torch.utils.data.DataLoader(test_feat_set, sampler=test_feat_sampler,
                                                    batch_size=args.batch_size)
    
    
    dist.barrier(group=group)
    if dist.get_rank() == 0:
        print("Feature dataloders created, training linear eval...") 
        
    linear_model = LinearClassifier(in_ftrs=2048, out_ftrs=2)
    linear_model = linear_model.cuda(gpu)
    linear_model = nn.parallel.DistributedDataParallel(linear_model, device_ids=[gpu])
    
    linear_criterion = nn.CrossEntropyLoss().cuda(gpu)
    linear_optimizer = torch.optim.Adam(linear_model.parameters(), lr=1e-3)
    
    epoch_losses = []
    epoch_accs = []
    # pretty similar to training loops for simclr
    for epoch in range(args.eval_epochs):
        if dist.get_rank() == 0:
            print(f"Epoch: {epoch + 1}")
            
        train_feat_sampler.set_epoch
        
        epoch_loss = 0
        epoch_acc = 0
        for batch, (x, label) in enumerate(train_feat_loader):
            x = x.cuda(gpu)
            label = label.long().cuda(gpu)
            
            output = linear_model(x_i)
            
            loss = linear_criterion(output, label)
            
            epoch_loss += loss
            
            loss.backward()
            linear_optimizer.step()
            
            predicted = output.argmax(1)
            acc = (predicted == label).sum().item() / label.size(0)
            epoch_acc += acc
                
        epoch_loss = epoch_loss.clone().detach()
        dist.reduce(epoch_loss, dst=0)
        
        epoch_acc = epoch_acc.clone().detach()
        dist.reduce(epoch_acc, dst=0)
        
        if dist.get_rank() == 0:
            mean_loss = epoch_loss.item() / (dist.get_world_size * len(train_feat_loader_loader))
            mean_acc = epoch_acc.item() / (dist.get_world_size * len(train_feat_loader))
            
            if epoch / 50 == 0:
                print("Epoch:", epoch,
                    "\tAverage Loss:", mean_loss,
                    "\tAverage Acc:", mean_acc)
            
            eval_losses.append(mean_loss)
            eval_accs.append(mean_acc)
            
    # once linear classifier is trained, evaluate on test features
    test_loss = 0
    test_acc = 0
    with torch.no_grad()
        for batch, (x, label) in enumerate(test_feat_loader):
                x = x.cuda(gpu)
                label = label.long().cuda(gpu)
                
                output = linear_model(x_i)
                
                loss = linear_criterion(output, label)
                
                test_loss += loss
                
                predicted = output.argmax(1)
                acc = (predicted == label).sum().item() / label.size(0)
                test_acc += acc
                    
            test_loss = test_loss.clone().detach()
            dist.reduce(test_loss, dst=0)
            
            test_acc = test_acc.clone().detach()
            dist.reduce(test_acc, dst=0)
            
            if dist.get_rank() == 0:
                mean_test_loss = test_loss.item() / (dist.get_world_size * len(test_feat_loader_loader))
                mean_test_acc = test_acc.item() / (dist.get_world_size * len(test_feat_loader))
                print("Test Set\t",
                    "Average Loss:", mean_test_loss,
                    "\tAverage Acc:", mean_test_acc)
    
    # save data at the end
    if dist.get_rank() == 0:            
        pretraining_losses = np.array(pretraining_losses)

        eval_losses = np.array(epoch_losses)
        eval_accs = np.array(epoch_accs)

        np_pt_path = os.path.join(data_path, "pretraining_data.npz")
        np.savez(np_pt_path, losses=pretrain_losses)
        
        np_eval_path = os.path.join(data_path, "eval_data.npz")
        np.savez(np.eval_path, losses=eval_losses, accs=eval_accs, test_loss=mean_test_loss, test_acc=mean_test_acc)

from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CosineAnnealingLR
from data import ModelNet40
from model import Pct
import numpy as np
from torch.utils.data import DataLoader
from util import cal_loss, IOStream
import sklearn.metrics as metrics
import torch.nn.functional as F
from torch.autograd import Variable
from sampler import ImbalancedDatasetSampler
import math
from munch import Munch
import time
import json
from unlabeled_sampler import Unlabeled_ImbalancedDatasetSampler

import time 

def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)
    if not os.path.exists('checkpoints/'+args.exp_name+'/'+'models'):
        os.makedirs('checkpoints/'+args.exp_name+'/'+'models')
    os.system('cp main.py checkpoints'+'/'+args.exp_name+'/'+'main.py.backup')
    os.system('cp model.py checkpoints' + '/' + args.exp_name + '/' + 'model.py.backup')
    os.system('cp util.py checkpoints' + '/' + args.exp_name + '/' + 'util.py.backup')
    os.system('cp data.py checkpoints' + '/' + args.exp_name + '/' + 'data.py.backup')




def train(args, io):
    train_loader_labeled = DataLoader(ModelNet40(partition='train', num_points=args.num_points, data_split='labeled', perceptange = 10), num_workers=8,
                            batch_size=args.batch_size, shuffle=True, drop_last=True)

    train_loader_unlabeled = DataLoader(ModelNet40(partition='train', num_points=args.num_points, data_split='unlabeled', perceptange = 10), num_workers=8,
                            batch_size=args.batch_size * args.unlabeled_ratio, shuffle=True, drop_last=True)

    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=8,
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)
    validate_loader = DataLoader(ModelNet40(partition='validate', num_points=args.num_points), num_workers=8,
                                     batch_size=args.test_batch_size, shuffle=False, drop_last=False)
    

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args).to(device)
    print(str(model))
    model = nn.DataParallel(model)
#     model_path = '/home//scratch1link/Semi-Vit/Semi/Semi-PCT_base500/checkpoints/train/models/latest_model.t7'
#     model.load_state_dict(torch.load(model_path))

    if args.use_sgd:
        print("Use SGD")
        opt = optim.SGD(model.parameters(), lr=args.lr*100, momentum=args.momentum, weight_decay=5e-4)
    else:
        print("Use Adam")
        opt = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)

    scheduler = CosineAnnealingLR(opt, args.epochs, eta_min=args.lr)

    criterion = cal_loss
    best_test_acc = 0
    

    section_interation = args.epochs / args.section_size    
    for current_section in range(int(section_interation)):
        np.random.seed()
        if current_section!=0:
            train_loader_labeled = DataLoader(
                ModelNet40(partition='train', num_points=args.num_points, data_split='labeled', perceptange=10),
                num_workers=8, sampler=ImbalancedDatasetSampler(
                    ModelNet40(partition='train', num_points=args.num_points, data_split='labeled', perceptange=10)),
                batch_size=args.batch_size, drop_last=True)
            
            train_loader_unlabeled = DataLoader(
                ModelNet40(partition='train', num_points=args.num_points, data_split='unlabeled', perceptange=10),
                num_workers=8, sampler=Unlabeled_ImbalancedDatasetSampler(
                    ModelNet40(partition='train', num_points=args.num_points, data_split='unlabeled', perceptange=10)),
                batch_size=args.batch_size*args.unlabeled_ratio, drop_last=True)
            

        for epoch in range(args.section_size):
            scheduler.step()
            train_loss = 0.0
            count = 0.0
            model.train()
            train_pred = []
            train_true = []
            idx = 0
            total_time = 0.0
            for l_data, u_data in zip(train_loader_labeled, train_loader_unlabeled):
                data_unaug, data, data_strongaug, label = l_data
                data_u_unaug, data_u, data_u_strongaug, label_u = u_data

                data_unaug, data, data_strongaug, label = data_unaug.to(device), data.to(device),data_strongaug.to(device), label.to(device).squeeze()
                data_u_unaug, data_u, data_u_strongaug, label_u = data_u_unaug.to(device), data_u.to(device), data_u_strongaug.to(device), label_u.to(device).squeeze()

                data_u_unaug = data_u_unaug.permute(0, 2, 1)
                data = data.permute(0, 2, 1)
                data_u = data_u.permute(0, 2, 1)
                data_strongaug = data_strongaug.permute(0, 2, 1)
                data_u_strongaug = data_u_strongaug.permute(0, 2, 1)

                batch_size = data.size()[0]
                opt.zero_grad()

                start_time = time.time()
                logits_l_w, tokens_l_w = model(data)
                logits_u_unaug, tokens_u_unaug = model(data_u_unaug)

                logits_u_s, tokens_u_s = model(data_u_strongaug)


                # logits, tokens1 = model(data)


                labeled_cross_entropy_loss = criterion(logits_l_w, label)


                pseudo_label = torch.softmax(logits_u_unaug.detach(), dim=-1)
                max_probs, targets_u = torch.max(pseudo_label, dim=-1)
                flex_threshold = torch.zeros(batch_size*args.unlabeled_ratio).to(device)

                
                if epoch + args.section_size*current_section == 0:
                    mask = max_probs.ge(0.2).float()
                else:
                    z = open("dict_avgconf.txt", "r")
                    k = z.read()
                    dict_avgconf = json.loads(k)
                    z.close()
    
                    for current_label in dict_avgconf:
                        current_label = int(current_label)
                
                        if dict_avgconf['%d' %current_label]/(2-dict_avgconf['%d' %current_label])<0.2:
                            flex_threshold[targets_u==current_label] = 0.2

                        elif dict_avgconf['%d' %current_label]/(2-dict_avgconf['%d' %current_label])>0.8:
                            flex_threshold[targets_u==current_label] = 0.8
                        else:
                            learning_effect = dict_avgconf['%d' %current_label]
                            flex_threshold[targets_u==current_label] = learning_effect/(2 - learning_effect)
                    mask = max_probs.ge(flex_threshold).float()                
                
                # print(valid_sample_num.item())
                mask_label = torch.ones(mask.shape[0])
                mask_label = Variable(mask_label).to('cuda')
                pt_pseudo_ce_loss = (F.cross_entropy(logits_u_s, targets_u, reduction='none') * mask).mean()


                loss = labeled_cross_entropy_loss + pt_pseudo_ce_loss
                loss.backward()
                opt.step()
                end_time = time.time()
                total_time += (end_time - start_time)

                preds = logits_l_w.max(dim=1)[1]
                count += batch_size
                train_loss += loss.item() * batch_size
                train_true.append(label.cpu().numpy())
                train_pred.append(preds.detach().cpu().numpy())
                idx += 1

            print ('train total time is',total_time)
            train_true = np.concatenate(train_true)
            train_pred = np.concatenate(train_pred)
            outstr = 'Train %d, loss: %.6f, train acc: %.6f, train avg acc: %.6f' % (epoch,
                                                                                    train_loss*1.0/count,
                                                                                    metrics.accuracy_score(
                                                                                    train_true, train_pred),
                                                                                    metrics.balanced_accuracy_score(
                                                                                    train_true, train_pred))
            io.cprint(outstr)

            ####################
            # Test
            ####################
            test_loss = 0.0
            count = 0.0
            model.eval()
            test_pred = []
            test_true = []
            total_time = 0.0
            for data, label in test_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                batch_size = data.size()[0]
                start_time = time.time()
                logits, tokens1 = model(data)
                end_time = time.time()
                total_time += (end_time - start_time)
                loss = criterion(logits, label)
                preds = logits.max(dim=1)[1]
                count += batch_size
                test_loss += loss.item() * batch_size
                test_true.append(label.cpu().numpy())
                test_pred.append(preds.detach().cpu().numpy())
            print ('test total time is', total_time)
            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_acc = metrics.accuracy_score(test_true, test_pred)
            avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
            outstr = 'Test %d, loss: %.6f, test acc: %.6f, test avg acc: %.6f' % (epoch + args.section_size*current_section ,
                                                                                test_loss*1.0/count,
                                                                                test_acc,
                                                                                avg_per_class_acc)
            io.cprint(outstr)
            if test_acc >= best_test_acc:
                best_test_acc = test_acc
                torch.save(model.state_dict(), 'checkpoints/%s/models/model.t7' % args.exp_name)
            torch.save(model.state_dict(), 'checkpoints/%s/models/latest_model.t7' % args.exp_name)

            
            test_true = []
            test_pred = []
            test_logits = []
            test_sec_max = []
            model.eval()
            for data, label in validate_loader:
                data, label = data.to(device), label.to(device).squeeze()
                data = data.permute(0, 2, 1)
                logits, tokens = model(data)
                m = nn.Softmax(dim=1)
                output = m(logits)
                sec_max, _ = torch.torch.sort(output, -1, descending=True)
                max_logits , preds  = output.max(dim=1)
                sec_max_logits = sec_max[:,1]
                if args.test_batch_size == 1:
                    test_true.append([label.cpu().numpy()])
                    test_pred.append([preds.detach().cpu().numpy()])
                    test_logits.append([max_logits.detach().cpu().numpy()])
                    test_sec_max.append([sec_max_logits.detach().cpu().numpy()])


                else:
                    test_true.append(label.cpu().numpy())
                    test_pred.append(preds.detach().cpu().numpy())
                    test_logits.append(max_logits.detach().cpu().numpy())
                    test_sec_max.append(sec_max_logits.detach().cpu().numpy())




            test_true = np.concatenate(test_true)
            test_pred = np.concatenate(test_pred)
            test_logits = np.concatenate(test_logits)
            print(test_logits.shape)

            labeled_data = np.arange(0,40)
            dict_1 = Munch()
            dict_secmax = Munch()
            dict_high_conf_index = Munch()
            dict_ave_validate_conf = Munch()

            for i in labeled_data:
                dict_1['%d' % i] = 0
                dict_secmax['%d' % i] = 0



            for current_label in labeled_data:
                current_label_index = np.where(test_pred == current_label)
                current_label_logit = test_logits[current_label_index]
                if len(current_label_index[0])!=0:
                    dict_ave_validate_conf['%d' % current_label] = np.sum(current_label_logit)/len(current_label_index[0])
                else:
                    dict_ave_validate_conf['%d' % current_label] = 0.2


            for current_label in range (40):
                label_pos = np.where(test_pred == current_label)

                if dict_ave_validate_conf['%d' % current_label]>0.8:
                    high_conf_label_pos = label_pos[0][np.where(test_logits[label_pos] > dict_ave_validate_conf['%d' % current_label])]
                    dict_high_conf_index['%d' % current_label] = high_conf_label_pos.tolist()



                elif dict_ave_validate_conf['%d' % current_label]<0.8:
                    low_conf_label_pos = label_pos[0][np.where(test_logits[label_pos] > dict_ave_validate_conf['%d' % current_label])]

                    dict_high_conf_index['%d_low' % current_label] = low_conf_label_pos.tolist()




            f = open("dict_highconf_indx.txt", "w")
            js = json.dumps(dict_high_conf_index)
            f.write(js)
            f.close()

            f = open("dict_avgconf.txt", "w")
            js = json.dumps(dict_ave_validate_conf)
            f.write(js)
            f.close()

            f = open("dict_logits.txt", "w")
            js = json.dumps(test_logits.tolist())
            f.write(js)
            f.close()

            f = open("current_epoch.txt", "w")
            js = json.dumps(epoch + args.section_size * current_section)
            f.write(js)
            f.close()
        

def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points),
                            batch_size=args.test_batch_size, shuffle=True, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")

    model = Pct(args).to(device)
    model = nn.DataParallel(model) 
    
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    test_true = []
    test_pred = []

    for data, label in test_loader:
        data, label = data.to(device), label.to(device).squeeze()
        data = data.permute(0, 2, 1)
        logits, tokens = model(data)
        preds = logits.max(dim=1)[1] 
        if args.test_batch_size == 1:
            test_true.append([label.cpu().numpy()])
            test_pred.append([preds.detach().cpu().numpy()])
        else:
            test_true.append(label.cpu().numpy())
            test_pred.append(preds.detach().cpu().numpy())




    test_true = np.concatenate(test_true)
    test_pred = np.concatenate(test_pred)
    test_acc = metrics.accuracy_score(test_true, test_pred)
    avg_per_class_acc = metrics.balanced_accuracy_score(test_true, test_pred)
    outstr = 'Test :: test acc: %.6f, test avg acc: %.6f'%(test_acc, avg_per_class_acc)
    io.cprint(outstr)

if __name__ == "__main__":
    # Training settings
    parser = argparse.ArgumentParser(description='Point Cloud Recognition')
    parser.add_argument('--section_size', type=int, default= 50, metavar='N',
                        help='how many epoches a section has ')
    parser.add_argument('--unlabeled_ratio', type=int, default= 4, metavar='N',
                        help='unlabeled labeled ratio ')
    parser.add_argument('--exp_name', type=str, default='train', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--dataset', type=str, default='modelnet40', metavar='N',
                        choices=['modelnet40'])
    parser.add_argument('--batch_size', type=int, default=32, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--epochs', type=int, default=250, metavar='N',
                        help='number of episode to train ')
    parser.add_argument('--use_sgd', type=bool, default=True,
                        help='Use SGD')
    parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                        help='learning rate (default: 0.001, 0.1 if using sgd)')
    parser.add_argument('--momentum', type=float, default=0.9, metavar='M',
                        help='SGD momentum (default: 0.9)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--eval', type=bool,  default=False,
                        help='evaluate the model')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/run.log')
    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint(
            'Using GPU : ' + str(torch.cuda.current_device()) + ' from ' + str(torch.cuda.device_count()) + ' devices')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    if not args.eval:
        train(args, io)
    else:
        test(args, io)

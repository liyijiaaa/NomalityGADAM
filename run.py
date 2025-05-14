import torch
import torch.nn as nn
import dgl
from dgl.data import register_data_args
import argparse, time
from model import *
from utils import *
from sklearn.metrics import roc_auc_score, recall_score, average_precision_score
from pytorch_memlab import LineProfiler, profile
#
from sklearn.preprocessing import MinMaxScaler
#


def train_local(net, graph, feats, opt, args, init=True):
    memo = {}
    labels = graph.ndata['label']
    num_nodes=  graph.num_nodes()

    device = args.gpu
    if device >= 0:
        torch.cuda.set_device(device)
        net = net.to(device)
        labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    if init:
        net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []
#！!!!!!!!!!!!!!!!!新加

    all_idx = list(range(num_nodes))
    memorybank = []#正太池
    # 创建一个形状为 (select_epoch,节点总数) 的全零矩阵，用于存储每个epoch中每个节点的异常分数。
    train_ano_score = torch.zeros((args.local_epochs, num_nodes), dtype=torch.float)

#
    for epoch in range(args.local_epochs):
        net.train()
        if epoch >= 3:
            t0 = time.time()

        opt.zero_grad()
        loss, l1, l2 = net(feats)

        loss.backward()
        opt.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(net.state_dict(), 'best_local_model.pkl')
 #
        pos = graph.ndata['pos']
        local_scores = -pos.detach()
        train_ano_score[epoch, all_idx] = local_scores
 #
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | l1 {:.4f} | l2 {:.4f}"
              .format(epoch+1, np.mean(dur), loss.item(), l1.item(), l2.item()))

#
        if epoch < args.local_epochs:
            # 选择上一轮异常分数低的节点（设置了k%）
            _, train_list_temp = train_ano_score[epoch - 1].topk(
            int( num_nodes- (epoch/args.local_epochs) ** 2 * num_nodes), dim=0,
            largest=False, sorted=True)
            train_list_temp = train_list_temp.cpu().numpy()
            train_list_temp = train_list_temp.tolist()
            memorybank.append(train_list_temp)

        if epoch == args.local_epochs:
            # 归一化处理
            train_ano_score = train_ano_score.cpu().detach().numpy()
            scaler = MinMaxScaler()
            train_ano_score = scaler.fit_transform(train_ano_score.T).T
            train_ano_score = torch.DoubleTensor(train_ano_score).cuda()

            # 计算每个节点在多个epoch中的平均异常得分
            for idx in range(len(memorybank)):
                train_ano_score[idx, memorybank[idx]] = 0
            train_ano_score_nonzero = torch.count_nonzero(train_ano_score, dim=0)
            train_ano_score = torch.sum(train_ano_score, dim=0)
            train_ano_score = train_ano_score / train_ano_score_nonzero

            # 选择平均异常分数最低的80 % 节点，假设这些是可靠的正样本
            _, train_list = train_ano_score.topk(int(0.80 * num_nodes), dim=0, largest=False, sorted=True)
            train_list = train_list.cpu().numpy()
            train_list = train_list.tolist()
            nor_idx = train_list
            net.nor_idx = nor_idx

    torch.save({'nor_idx': nor_idx}, 'node_indices.pth')
    memo['graph'] = graph
    net.load_state_dict(torch.load('best_local_model.pkl'))
    h, mean_h = net.encoder(feats)
    h, mean_h = h.detach(), mean_h.detach()
    memo['h'] = h
    memo['mean_h'] = mean_h

    torch.save(memo, 'memo.pth')


def load_info_from_local(local_net, device):
    if device >= 0:
        torch.cuda.set_device(device)
        local_net = local_net.to(device)

    memo = torch.load('memo.pth')
    local_net.load_state_dict(torch.load('best_local_model.pkl'))
    graph = memo['graph']

    pos = graph.ndata['pos']

    scores = -pos.detach()


    #要对这部分的代码进行修改
    ano_topk = 0.05  # k_ano
    num_nodes = graph.num_nodes()

    num_ano = int(num_nodes * ano_topk)
    _, ano_idx = torch.topk(scores, num_ano)
#
    indices = torch.load('node_indices.pth')
    nor_idx = indices['nor_idx']
#
    feats = graph.ndata['feat']

    h, _ = local_net.encoder(feats)

    center = h[nor_idx].mean(dim=0).detach()

    if device >= 0:
        memo = {k: v.to(device) for k, v in memo.items()}
        nor_idx = nor_idx.cuda()
        ano_idx = ano_idx.cuda()
        center = center.cuda()

    return memo, nor_idx, ano_idx, center


def train_global(global_net, opt, graph, args):
    epochs = args.global_epochs

    labels = graph.ndata['label'].cpu().numpy()
    num_nodes=  graph.num_nodes()
    device = args.gpu
    feats = graph.ndata['feat']
    pos = graph.ndata['pos']

    if device >= 0:
        torch.cuda.set_device(device)
        global_net = global_net.to(device)
        # labels = labels.cuda()
        feats = feats.cuda()

    def init_xavier(m):
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)
    
    init = True
    if init:
        global_net.apply(init_xavier)
    
    print('train on:', 'cpu' if device<0 else 'gpu {}'.format(device))

    cnt_wait = 0
    best = 999
    dur = []

    pred_labels = np.zeros_like(labels)
    for epoch in range(epochs):
        global_net.train()
        if epoch >= 3:
            t0 = time.time()

        opt.zero_grad()
        loss, scores = global_net(feats, epoch)
        loss.backward()
        opt.step()

        if epoch >= 3:
            dur.append(time.time() - t0)

        if loss.item() < best:
            best = loss.item()
            torch.save(global_net.state_dict(), 'best_global_model.pkl')

        mix_score = -(scores + pos)
        mix_score = mix_score.detach().cpu().numpy()

        mix_auc = roc_auc_score(labels, mix_score)
        
        sorted_idx = np.argsort(mix_score)
        k = int(sum(labels))
        topk_idx = sorted_idx[-k:]
        pred_labels[topk_idx] = 1

        recall_k = recall_score(np.ones(k), labels[topk_idx])
        ap = average_precision_score(labels, mix_score)

        # print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | auc {:.4f} | mix_auc {:.4f}"
        #       .format(epoch+1, np.mean(dur), loss.item(), auc, mix_auc))
        print("Epoch {} | Time(s) {:.4f} | Loss {:.4f} | mix_auc {:.4f} | recall@k {:.4f} | ap {:.4f}"
            .format(epoch+1, np.mean(dur), loss.item(), mix_auc, recall_k, ap))
    
    return mix_auc, recall_k, ap

def main(args):
    seed_everything(args.seed)

    graph = my_load_data(args.data)
    # graph = graph.add_self_loop() test encoder=GCN
    feats = graph.ndata['feat']

    if args.gpu >= 0:
        graph = graph.to(args.gpu)

    in_feats = feats.shape[1]#获取节点特征的长度，维度

    local_net = LocalModel(graph,
                     in_feats,
                     args.out_dim,
                     nn.PReLU(),)

    local_opt = torch.optim.Adam(local_net.parameters(), 
                                 lr=args.local_lr, 
                                 weight_decay=args.weight_decay)
    t1 = time.time()
    train_local(local_net, graph, feats, local_opt, args)
    
    # load information from LIM module
    memo, nor_idx, ano_idx, center = load_info_from_local(local_net, args.gpu)
    t2 = time.time()
    graph = memo['graph']
    global_net = GlobalModel(graph, 
                             in_feats, 
                             args.out_dim, 
                             nn.PReLU(), 
                             nor_idx, 
                             ano_idx, 
                             center)
    opt = torch.optim.Adam(global_net.parameters(), 
                                 lr=args.global_lr, 
                                 weight_decay=args.weight_decay)
    t3 = time.time()
    
    mix_auc, recall_k, ap = train_global(global_net, opt, graph, args)
    t4 = time.time()

    t_all = t2+t4-t1-t3
    print('mean_t:{:.4f}'.format(t_all / (args.local_epochs + args.global_epochs)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='model')
    register_data_args(parser)
    parser.add_argument("--data", type=str, default="Cora",
                        help="dataset")
    parser.add_argument("--seed", type=int, default=717,
                        help="random seed")
    parser.add_argument("--dropout", type=float, default=0.,
                        help="dropout probability")
    parser.add_argument("--gpu", type=int, default=1,
                        help="gpu")
    parser.add_argument("--local-lr", type=float, default=1e-3,
                        help="learning rate")
    parser.add_argument("--global-lr", type=float, default=5e-4,
                        help="learning rate")
    parser.add_argument("--local-epochs", type=int, default=100,
                        help="number of training local model epochs")
    parser.add_argument("--global-epochs", type=int, default=50,
                        help="number of training global model epochs")
    parser.add_argument("--out-dim", type=int, default=64,
                        help="number of hidden gcn units")
    parser.add_argument("--train-ratio", type=float, default=0.05,
                        help="train ratio")
    parser.add_argument("--weight-decay", type=float, default=0.,
                        help="Weight for L2 loss")
    parser.add_argument("--patience", type=int, default=20,
                        help="early stop patience condition")
    parser.add_argument("--self-loop", action='store_true',
                        help="graph self-loop (default=False)")
    parser.set_defaults(self_loop=True)

    args = parser.parse_args()
    print(args)
    main(args)
    # multi_run(args)

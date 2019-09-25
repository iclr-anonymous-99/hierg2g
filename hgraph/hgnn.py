import torch
import torch.nn as nn
import rdkit.Chem as Chem
import torch.nn.functional as F
from mol_graph import MolGraph
from encoder import HierMPNEncoder
from decoder import HierMPNDecoder
from nnutils import *

def make_cuda(tensors):
    tree_tensors, graph_tensors = tensors
    tree_tensors = [x.cuda().long() for x in tree_tensors[:-1]] + [tree_tensors[-1]]
    graph_tensors = [x.cuda().long() for x in graph_tensors[:-1]] + [graph_tensors[-1]]
    return tree_tensors, graph_tensors

class HierVAE(nn.Module):

    def __init__(self, args):
        super(HierVAE, self).__init__()
        self.encoder = HierMPNEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depthT, args.depthG, args.dropout)
        self.decoder = HierMPNDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.latent_size, args.diterT, args.diterG, args.dropout)
        self.encoder.tie_embedding(self.decoder.hmpn)
        self.T_mean = nn.Linear(args.hidden_size * 2, args.latent_size)
        self.T_var = nn.Linear(args.hidden_size * 2, args.latent_size)
        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.G_var = nn.Linear(args.hidden_size, args.latent_size)

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def encode(self, graphs, tensors):
        tree_batch, graph_batch = graphs
        tree_tensors, graph_tensors = make_cuda(tensors)
        root_vecs, tree_vecs, inter_vecs, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        graph_vecs = self.anchor_pooling(tree_batch, graph_tensors, graph_vecs)

        tree_vecs = stack_pad_tensor( [tree_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        size = tree_vecs.new_tensor([le for _,le in tree_tensors[-1]])
        tree_vecs = tree_vecs.sum(dim=1) / size.unsqueeze(-1)

        inter_vecs = stack_pad_tensor( [inter_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        inter_vecs = inter_vecs.sum(dim=1) / size.unsqueeze(-1)

        tree_vecs = torch.cat( [root_vecs, tree_vecs, inter_vecs], dim=-1 )
        tree_vecs = self.T_mean(tree_vecs)
        graph_vecs = self.G_mean(graph_vecs)
        return tree_vecs, tree_vecs, graph_vecs #tree_vecs is root_vecs

    def decode(self, mol_vecs):
        return self.decoder.decode(mol_vecs)
        
    def forward(self, graphs, tensors, orders, beta):
        tree_tensors, graph_tensors = make_cuda(tensors)
        tree_batch, graph_batch = graphs
        tensors = (tree_tensors, graph_tensors)

        root_vecs, tree_vecs, inter_vecs, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        graph_vecs = self.anchor_pooling(tree_batch, graph_tensors, graph_vecs)

        tree_vecs = stack_pad_tensor( [tree_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        size = tree_vecs.new_tensor([le for _,le in tree_tensors[-1]])
        tree_vecs = tree_vecs.sum(dim=1) / size.unsqueeze(-1)

        inter_vecs = stack_pad_tensor( [inter_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        inter_vecs = inter_vecs.sum(dim=1) / size.unsqueeze(-1)

        tree_vecs = torch.cat( [root_vecs, tree_vecs, inter_vecs], dim=-1 )

        tree_vecs, tree_kl = self.rsample(tree_vecs, self.T_mean, self.T_var)
        graph_vecs, graph_kl = self.rsample(graph_vecs, self.G_mean, self.G_var)
        kl_div = tree_kl + graph_kl

        loss, wacc, iacc, tacc, sacc = self.decoder((tree_vecs, tree_vecs, graph_vecs), graphs, tensors, orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc

    def anchor_pooling(self, tree_batch, graph_tensors, graph_vecs):
        anchors = []
        for x,attr in tree_batch.nodes(data=True):
            if len(attr['assm_cands']) == 0: continue
            anchors.extend( zip(*attr['inter_label'])[0] )

        anchors = set(anchors)
        pool_graph_vecs = []
        for st,le in graph_tensors[-1]:
            cur_anchors = [i for i in range(st, st + le) if i in anchors]
            if len(cur_anchors) == 0: cur_anchors = [0]
            cur_anchors = graph_tensors[0].new_tensor(cur_anchors)
            cur_anchor_vecs = graph_vecs.index_select(0, cur_anchors)
            pool_graph_vecs.append( cur_anchor_vecs.sum(dim=0) )

        return torch.stack(pool_graph_vecs, dim=0)


class HierVGNN(nn.Module):

    def __init__(self, args):
        super(HierVGNN, self).__init__()
        self.latent_size = args.latent_size
        self.encoder = HierMPNEncoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.depthT, args.depthG, args.dropout)
        self.decoder = HierMPNDecoder(args.vocab, args.atom_vocab, args.rnn_type, args.embed_size, args.hidden_size, args.hidden_size, args.diterT, args.diterG, args.dropout, attention=True)
        self.encoder.tie_embedding(self.decoder.hmpn)

        self.T_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.T_var = nn.Linear(args.hidden_size, args.latent_size)
        self.G_mean = nn.Linear(args.hidden_size, args.latent_size)
        self.G_var = nn.Linear(args.hidden_size, args.latent_size)

        self.W_tree = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size, args.hidden_size), nn.ReLU() )
        self.W_graph = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size, args.hidden_size), nn.ReLU() )

    def encode(self, tensors):
        tree_tensors, graph_tensors = tensors
        root_vecs, tree_vecs, _, graph_vecs = self.encoder(tree_tensors, graph_tensors)
        tree_vecs = stack_pad_tensor( [tree_vecs[st : st + le] for st,le in tree_tensors[-1]] )
        graph_vecs = stack_pad_tensor( [graph_vecs[st : st + le] for st,le in graph_tensors[-1]] )
        return root_vecs, tree_vecs, graph_vecs

    def translate(self, tensors, num_decode, enum_root):
        tensors = make_cuda(tensors)
        root_vecs, tree_vecs, graph_vecs = self.encode(tensors)
        all_smiles = []
        if enum_root:
            num_root = len(root_vecs)
            all_root_vecs, all_tree_vecs, all_graph_vecs = root_vecs, tree_vecs, graph_vecs
            for i in xrange(num_decode):
                root_vecs = all_root_vecs[i % num_root].unsqueeze(0)
                tree_vecs = all_tree_vecs[i % num_root].unsqueeze(0)
                graph_vecs = all_graph_vecs[i % num_root].unsqueeze(0)
                z_tree = torch.randn(1, 1, self.latent_size).expand(-1, tree_vecs.size(1), -1).cuda()
                z_graph = torch.randn(1, 1, self.latent_size).expand(-1, graph_vecs.size(1), -1).cuda()
                z_tree_vecs = self.W_tree( torch.cat([tree_vecs, z_tree], dim=-1) )
                z_graph_vecs = self.W_graph( torch.cat([graph_vecs, z_graph], dim=-1) )
                smiles = self.decoder.decode( (root_vecs, z_tree_vecs, z_graph_vecs) )
                all_smiles.append(smiles)
        else:
            batch_size = len(root_vecs)
            for i in xrange(num_decode):
                z_tree = torch.randn(batch_size, 1, self.latent_size).expand(-1, tree_vecs.size(1), -1).cuda()
                z_graph = torch.randn(batch_size, 1, self.latent_size).expand(-1, graph_vecs.size(1), -1).cuda()
                z_tree_vecs = self.W_tree( torch.cat([tree_vecs, z_tree], dim=-1) )
                z_graph_vecs = self.W_graph( torch.cat([graph_vecs, z_graph], dim=-1) )
                smiles = self.decoder.decode( (root_vecs, z_tree_vecs, z_graph_vecs) )
                all_smiles.append(smiles)

        return all_smiles

    def rsample(self, z_vecs, W_mean, W_var):
        batch_size = z_vecs.size(0)
        z_mean = W_mean(z_vecs)
        z_log_var = -torch.abs( W_var(z_vecs) )
        kl_loss = -0.5 * torch.sum(1.0 + z_log_var - z_mean * z_mean - torch.exp(z_log_var)) / batch_size
        epsilon = torch.randn_like(z_mean).cuda()
        z_vecs = z_mean + torch.exp(z_log_var / 2) * epsilon
        return z_vecs, kl_loss

    def forward(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders, beta):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        x_root_vecs, x_tree_vecs, x_graph_vecs = self.encode(x_tensors)
        _, y_tree_vecs, y_graph_vecs = self.encode(y_tensors)

        diff_tree_vecs = y_tree_vecs.sum(dim=1) - x_tree_vecs.sum(dim=1)
        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_tree_vecs, tree_kl = self.rsample(diff_tree_vecs, self.T_mean, self.T_var)
        diff_graph_vecs, graph_kl = self.rsample(diff_graph_vecs, self.G_mean, self.G_var)
        kl_div = tree_kl + graph_kl

        diff_tree_vecs = diff_tree_vecs.unsqueeze(1).expand(-1, x_tree_vecs.size(1), -1)
        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_tree_vecs = self.W_tree( torch.cat([x_tree_vecs, diff_tree_vecs], dim=-1) )
        x_graph_vecs = self.W_graph( torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1) )

        loss, wacc, iacc, tacc, sacc = self.decoder((x_root_vecs, x_tree_vecs, x_graph_vecs), y_graphs, y_tensors, y_orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc

class HierCondVGNN(HierVGNN):

    def __init__(self, args):
        super(HierCondVGNN, self).__init__(args)
        self.W_tree = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size + args.cond_size, args.hidden_size), nn.ReLU() )
        self.W_graph = nn.Sequential( nn.Linear(args.hidden_size + args.latent_size + args.cond_size, args.hidden_size), nn.ReLU() )

        self.U_tree = nn.Sequential( nn.Linear(args.hidden_size + args.cond_size, args.hidden_size), nn.ReLU() )
        self.U_graph = nn.Sequential( nn.Linear(args.hidden_size + args.cond_size, args.hidden_size), nn.ReLU() )

    def translate(self, tensors, cond, num_decode, enum_root):
        assert enum_root 
        tensors = make_cuda(tensors)
        root_vecs, tree_vecs, graph_vecs = self.encode(tensors)

        cond = cond.view(1,1,-1)
        tree_cond = cond.expand(-1, tree_vecs.size(1), -1)
        graph_cond = cond.expand(-1, graph_vecs.size(1), -1)

        all_root_vecs, all_tree_vecs, all_graph_vecs = root_vecs, tree_vecs, graph_vecs
        all_smiles = []
        num_root = len(root_vecs)

        for i in xrange(num_decode):
            root_vecs = all_root_vecs[i % num_root].unsqueeze(0)
            tree_vecs = all_tree_vecs[i % num_root].unsqueeze(0)
            graph_vecs = all_graph_vecs[i % num_root].unsqueeze(0)
            z_tree = torch.randn(1, 1, self.latent_size).expand(-1, tree_vecs.size(1), -1).cuda()
            z_graph = torch.randn(1, 1, self.latent_size).expand(-1, graph_vecs.size(1), -1).cuda()
            z_tree_vecs = self.W_tree( torch.cat([tree_vecs, z_tree, tree_cond], dim=-1) )
            z_graph_vecs = self.W_graph( torch.cat([graph_vecs, z_graph, graph_cond], dim=-1) )
            smiles = self.decoder.decode( (root_vecs, z_tree_vecs, z_graph_vecs) )
            all_smiles.append(smiles)
    
        return all_smiles

    def forward(self, x_graphs, x_tensors, y_graphs, y_tensors, y_orders, cond, beta):
        x_tensors = make_cuda(x_tensors)
        y_tensors = make_cuda(y_tensors)
        cond = cond.float().cuda()

        x_root_vecs, x_tree_vecs, x_graph_vecs = self.encode(x_tensors)
        _, y_tree_vecs, y_graph_vecs = self.encode(y_tensors)

        diff_tree_vecs = y_tree_vecs.sum(dim=1) - x_tree_vecs.sum(dim=1)
        diff_graph_vecs = y_graph_vecs.sum(dim=1) - x_graph_vecs.sum(dim=1)
        diff_tree_vecs = self.U_tree( torch.cat([diff_tree_vecs, cond], dim=-1) ) #combine condition for posterior
        diff_graph_vecs = self.U_graph( torch.cat([diff_graph_vecs, cond], dim=-1) ) #combine condition for posterior

        diff_tree_vecs, tree_kl = self.rsample(diff_tree_vecs, self.T_mean, self.T_var)
        diff_graph_vecs, graph_kl = self.rsample(diff_graph_vecs, self.G_mean, self.G_var)
        kl_div = tree_kl + graph_kl

        diff_tree_vecs = torch.cat([diff_tree_vecs, cond], dim=-1) #combine condition for posterior
        diff_graph_vecs = torch.cat([diff_graph_vecs, cond], dim=-1) #combine condition for posterior

        diff_tree_vecs = diff_tree_vecs.unsqueeze(1).expand(-1, x_tree_vecs.size(1), -1)
        diff_graph_vecs = diff_graph_vecs.unsqueeze(1).expand(-1, x_graph_vecs.size(1), -1)
        x_tree_vecs = self.W_tree( torch.cat([x_tree_vecs, diff_tree_vecs], dim=-1) )
        x_graph_vecs = self.W_graph( torch.cat([x_graph_vecs, diff_graph_vecs], dim=-1) )

        loss, wacc, iacc, tacc, sacc = self.decoder((x_root_vecs, x_tree_vecs, x_graph_vecs), y_graphs, y_tensors, y_orders)
        return loss + beta * kl_div, kl_div.item(), wacc, iacc, tacc, sacc


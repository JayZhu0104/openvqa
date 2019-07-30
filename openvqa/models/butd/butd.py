# --------------------------------------------------------
# OpenVQA
# Written by Yuhao Cui https://github.com/cuiyuhao1996
# --------------------------------------------------------


from openvqa.ops.layer_norm import LayerNorm

import torch.nn as nn
import torch.nn.functional as F
import torch, math
from torch.nn.utils.weight_norm import weight_norm



#------------------------------
# ----Fully Connect Network----
# ------------------------------

class FCNet(nn.Module):
    """Simple class for non-linear fully connect network
    """
    def __init__(self, dims):
        super(FCNet, self).__init__()

        layers = []
        for i in range(len(dims)-2):
            in_dim = dims[i]
            out_dim = dims[i+1]
            layers.append(weight_norm(nn.Linear(in_dim, out_dim), dim=None))
            layers.append(nn.ReLU())
        layers.append(weight_norm(nn.Linear(dims[-2], dims[-1]), dim=None))
        layers.append(nn.ReLU())

        self.main = nn.Sequential(*layers)

    def forward(self, x):
        return self.main(x)


#------------------------------
# ---- Attention ----
# ------------------------------

class Attention(nn.Module):
    def __init__(self, __C):
        super(Attention, self).__init__()
        self.nonlinear = FCNet([__C.HIDDEN_SIZE + __C.FF_SIZE, __C.HIDDEN_SIZE])
        self.linear = weight_norm(nn.Linear(__C.HIDDEN_SIZE, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        num_objs = v.size(1)
        q = q.unsqueeze(1).repeat(1, num_objs, 1)
        vq = torch.cat((v, q), 2)
        joint_repr = self.nonlinear(vq)
        logits = self.linear(joint_repr)
        return logits

#------------------------------
# ---- NewAttention ----
# ------------------------------

class NewAttention(nn.Module):
    def __init__(self, __C):
        super(NewAttention, self).__init__()

        self.__C = __C
        #这里的参数名还没确定，只是把model_cfgs中等值的参数名传了过去
        self.v_proj = FCNet([__C.HIDDEN_SIZE, __C.HIDDEN_SIZE])
        self.q_proj = FCNet([__C.FF_SIZE, __C.HIDDEN_SIZE])
        self.dropout = nn.Dropout(__C.DROPOUT_R)
        #下面的__C.FF_SIZE还不确定
        self.linear = weight_norm(nn.Linear(__C.FF_SIZE, 1), dim=None)

    def forward(self, v, q):
        """
        v: [batch, k, vdim]
        q: [batch, qdim]
        """
        logits = self.logits(v, q)
        #用sigmoid
        w = nn.functional.softmax(logits, 1)
        return w

    def logits(self, v, q):
        batch, k, _ = v.size()
        v_proj = self.v_proj(v) # [batch, k, qdim]
        q_proj = self.q_proj(q).unsqueeze(1).repeat(1, k, 1)
        joint_repr = v_proj * q_proj
        #没做Relu
        joint_repr = self.dropout(joint_repr)
        logits = self.linear(joint_repr)
        return logits


#------------------------------
# ---- BaseModel ----
# ------------------------------

class BM(nn.Module):
    def __init__(self, q_emb, v_att, q_net, v_net):
    # 这几个参数还没明白
    # def __init__(self, __C):
        super(BM, self).__init__()
        # self.w_emb = __C.WORD_EMBED_SIZE
        self.q_emb = q_emb
        self.v_att = v_att
        self.q_net = q_net
        self.v_net = v_net
        # self.classifier = classifier

    def forward(self, v, b, q, labels):
        """Forward

        v: [batch, num_objs, obj_dim]
        b: [batch, num_objs, b_dim]
        q: [batch_size, seq_length]

        return: logits, not probs
        """
        # w_emb = self.w_emb(q)
        q_emb = self.q_emb() # [batch, q_dim]

        att = self.v_att(v, q_emb)
        v_emb = (att * v).sum(1) # [batch, v_dim]

        q_repr = self.q_net(q_emb)
        v_repr = self.v_net(v_emb)
        joint_repr = q_repr * v_repr
        logits = self.classifier(joint_repr)
        return logits


#------------------------------
# ---- Bottom Up Top Down ----
# ------------------------------

class BUTD(nn.Module):
    def __init__(self, __C):
        super(BUTD, self).__init__()
        self.newattention = NewAttention(__C)
        self.q_net = FCNet([__C.HIDDEN_SIZE, __C.HIDDEN_SIZE])
        self.v_net = FCNet([__C.FF_SIZE, __C.HIDDEN_SIZE])
    def forward(self, x, y):
        v_att = self.newattention(y, x)
        q_net = self.q_net()
        v_net = self.v_net()
        return BM(x, v_att, q_net, v_net)
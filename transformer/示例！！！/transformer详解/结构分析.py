'''一、Encoder
1. embedding:
    input=**X=[batchsize,sequence_length]**
    经过input embedding=**Xembedding=[batchsize,sequence_length,embedding dimension]** **这里维数？？？？**
    比如输入车辆历史信息[256,60] →[256,60,2]

    本质：从高维稀疏sparse到低维稠密dense
        李宏毅第5课的word embedding中从one-hot vector(比如每个词1*1e6维) 到  word vector (1e3*1e3的matrix)

2. positional encoding（/embedding）:
看这个https://wmathor.com/index.php/archives/1453/
    1) 因为是并行，不像lstm是串行，所以要理解输入X的成员顺序，要加这个
        位置嵌入向量维数**[max_sequence_length, embedding_dimension]**，max_sequence_length 属于超参数，指的是限定每个句子最长由多少个词构成
        比如历史信息最长70，所以[70,2]
    2) 以字为单位训练 Transformer 模型
        vocab_size 为字库中所有字的数量(输入多少辆车，就有多少条轨迹)，embedding_dimension 为字向量的维度
        首先初始化字编码nn.Embedding(vocab_size, embedding_dimension)'''
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import math

# def get_input_embedding():
#  #字嵌入???

def get_positional_encoding(max_seq_len,embed_dim): 
    #位置嵌入，和字嵌入piece-wise对应位置相加作为输入
    # # 初始化一个positional encoding
    # embed_dim: 字嵌入的维度
    # max_seq_len: 字库最大的序列长度
    positional_encoding=np.array([
        [pos/np.power(1000,2*i/embed_dim) for i in range(embed_dim)] \
            if pos!=0 else np.zeros(embed_dim) for pos in range(max_seq_len)
    ])

    positional_encoding[1:, 0::2] = np.sin(positional_encoding[1:, 0::2])  # 当dim 2i 偶数
    positional_encoding[1:, 1::2] = np.cos(positional_encoding[1:, 1::2])  # 当dim 2i+1 奇数
    return positional_encoding
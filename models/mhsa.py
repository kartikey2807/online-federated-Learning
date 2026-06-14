from torchsummary import summary

import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class InputEmbedding(nn.Module):
    '''
    Converts input matrix to compatible embeddings to add with position encoding.\n
    We use a linear layer for continuous tabular data.\n
    Parameters:
    ----------
    1. input_size: number of features in matrix.
    2. embedding_dimension: dimensions
    '''
    def __init__(self,input_size:int,embedding_dimension:int) -> None:
        super().__init__()

        self.fc01 = nn.Linear(input_size,embedding_dimension)
        self.scaled = math.sqrt(embedding_dimension)
        nn.init.xavier_uniform_(self.fc01.weight)

    def forward(self,input:torch.Tensor) -> torch.Tensor:
        return self.fc01(input) * self.scaled

class PositionalEncoding(nn.Module):
    '''
    Learnable embedding that gives a vector to each position in the input vector.\n
    It works in cases where window_size is fixed.\n
    Parameters:
    ----------
    1. embedding_dimension: dimensions
    2. max_length: maximum window length (fixed)
    '''
    def __init__(self,embedding_dimension:int,max_length:int) -> None:
        super().__init__()

        self.embed = nn.Embedding(max_length,embedding_dimension)
        nn.init.normal_(self.embed.weight,std=0.02)
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        positions = torch.arange(0,input.size(1),device=input.device).unsqueeze(0)
        return input + self.embed(positions)

class MultiHeadSelfAttention(nn.Module):
    '''
    Multi-head self attention blocks.\n
    Parameters:
    ----------
    1. embedding_dimension: dimensions
    2. num_heads: number of parallel head for multiplication.
    '''
    def __init__(self,embedding_dimension:int,num_heads:int):
        super().__init__()

        assert (embedding_dimension%num_heads == 0), 'embedding dimension should be divisible by number of heads'
        self.embedding_dimension = embedding_dimension
        self.num_heads = num_heads
        self.dk = embedding_dimension // num_heads
        self.scaled = math.sqrt(self.dk)

        self.K = nn.Linear(embedding_dimension,embedding_dimension)
        self.Q = nn.Linear(embedding_dimension,embedding_dimension)
        self.V = nn.Linear(embedding_dimension,embedding_dimension)
        self.o = nn.Linear(embedding_dimension,embedding_dimension)

        ## initialization
        for layers in [self.K, self.Q, self.V, self.o]:
            nn.init.xavier_uniform_(layers.weight)
            nn.init.zeros_(layers.bias)
    
    def split_heads(self,input:torch.Tensor) -> torch.Tensor:

        B,L,_ = input.shape
        input = input.view(B,L,self.num_heads,self.dk)
        return input.transpose(1,2)

    def forward(self,input:torch.Tensor) -> torch.Tensor:

        Eq = self.split_heads(self.Q(input))
        Ek = self.split_heads(self.K(input))
        Ev = self.split_heads(self.V(input))

        attention_map = F.softmax(torch.matmul(Eq,Ek.transpose(-2,-1))/self.scaled,dim=-1)
        output = torch.matmul(attention_map,Ev)

        B,H,L,_ = output.shape
        output = output.transpose(1,2).contiguous()
        output = output.view(B,L,self.embedding_dimension)
        return self.o(output),attention_map

class MLP(nn.Module):
    '''
    Feed forward network with bottleneck layer.\n
    Parameters:
    ----------
    1. embedding_dimension: dimensions
    2. Dimensions for the bottleneck layers will be 4 times embedding dimensions.
    '''
    def __init__(self,embedding_dimension:int) -> None:
        super().__init__()

        self.fc01 = nn.Linear(embedding_dimension,4*embedding_dimension)
        self.fc02 = nn.Linear(4*embedding_dimension,embedding_dimension)
        self.dropout = nn.Dropout(p=0.1)
        self.activation = nn.GELU()

        nn.init.kaiming_uniform_(self.fc01.weight,nonlinearity = 'relu')
        nn.init.xavier_uniform_(self.fc02.weight)
        for layer in [self.fc01,self.fc02]:
            nn.init.zeros_(layer.bias)
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        x = self.fc01(input)
        x = self.activation(x)
        x = self.dropout(x)
        x = self.fc02(x)
        return x

class TransformerBlock(nn.Module):
    '''
    pre-layernorm for stable training\n
    x = x + MHSA(LN(x))\n
    x = x + FFNW(LN(x))\n
    Parameters:
    ----------
    1. embedding_dimension: dimensions
    2. num_heads: number of MHSA heads
    3. input_size: numbers of features
    4. max_length: window length
    '''
    def __init__(self,embedding_dimension:int,num_heads:int,input_size:int,max_length:int) -> None:
        super().__init__()

        self.MHSA = MultiHeadSelfAttention(embedding_dimension,num_heads)
        self.FFNW = MLP(embedding_dimension)
        self.norm1 = nn.LayerNorm(embedding_dimension)
        self.norm2 = nn.LayerNorm(embedding_dimension)
        self.dropout = nn.Dropout(p=0.1)
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:

        ## Pre-LayerNorm by design

        x1 = input
        x2,attention_map = self.MHSA(self.norm1(x1))
        x3 = input + self.dropout(x2)

        x4 = x3
        x5 = self.FFNW(self.norm2(x4))
        x6 = x3 + self.dropout(x5)
        return x6,attention_map
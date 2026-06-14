from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.mhsa import TransformerBlock, InputEmbedding, PositionalEncoding ## helpers

class Encoder(nn.Module):
    '''
    Encoder of stacked transformer blocks.\n
    Input Embedding (B,L,D)\n
    -> N x Transformer Blocks (B,L,D)\n
    -> Final LayerNorm\n
    -> Bottleneck Linear (B,L,bottle_neck)\n
    Parameters:
    ----------
    1. embedding_dimension: dimensions 
    2. num_heads: number of MSHA heads
    3. input_size: numbers of features
    4. max_length: WINDOW_LENGTH
    5. num_layers: number of stacked Transformer blocks
    6. bottleneck_dim: dimension for bottlenck
    '''
    def __init__(
            self,
            embedding_dimension:int,
            num_heads:int,
            input_size:int,
            max_length:int,
            num_layers:int,
            bottleneck_dim:int,
        ):
        super().__init__()

        self.layers = nn.ModuleList([
            TransformerBlock(embedding_dimension,num_heads,input_size,max_length)
            for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(
            embedding_dimension
        )
        ## ------------------- Autoencoder Part -------------------
        # self.bottleneck = nn.Sequential(
        #     nn.Linear(embedding_dimension,bottleneck_dim),
        #     nn.GELU()
        # )
        ## --------------------------------------------------------
        
        self.mu     = nn.Linear(embedding_dimension,bottleneck_dim)
        self.logvar = nn.Linear(embedding_dimension,bottleneck_dim)
    
    def reparameterization(self,mu,logvar):

        if self.training:
            std = torch.exp(0.5*logvar)
            eps = torch.randn_like(std)
            return mu + (std*eps)
        return mu

    def forward(self,input:torch.Tensor):

        attention_map_record = []
        for layer in self.layers:
            input,attention_map = layer(input)
            attention_map_record.append(attention_map)
        
        input = self.layernorm(input)
        temp_mu = self.mu(input)
        temp_logvar = self.logvar(input)
        latent_tensors  = self.reparameterization(temp_mu,temp_logvar)
        return latent_tensors,temp_mu,temp_logvar,attention_map_record

class Decoder(nn.Module):
    '''
    Decoder of stacked transformer blocks.\n
    Latent (B,L,bottle_neck)\n
    -> Linear (B,L,D)\n
    -> N x Transformer blocks (B,L,D)\n
    -> Final Layer Norm\n
    -> Output Projection (B,L,input_size)\n
    Parameters:
    ----------
    1. embedding_dimension: dimensions
    2. num_heads: number of MSHA heads
    3. num_layers: number of Transformer layers
    4. bottleneck_dim: bottleneck
    5. input_size: numbers of features
    6. max_length: WINDOW_LENGTH
    '''
    def __init__(
            self,
            embedding_dimension:int,
            num_heads:int,
            num_layers:int,
            input_size:int,
            max_length:int,
            bottleneck_dim:int,
        ):

        super().__init__()

        self.expand = nn.Sequential(
            nn.Linear(bottleneck_dim,embedding_dimension),
            nn.GELU()
        )
        self.layers = nn.ModuleList([
            TransformerBlock(embedding_dimension,num_heads,input_size,max_length)
            for _ in range(num_layers)
        ])
        self.layernorm = nn.LayerNorm(embedding_dimension)
        self.projection = nn.Linear(embedding_dimension,input_size)
    
    def forward(self,bottleneck:torch.Tensor):
        h = self.expand(bottleneck)
        
        attention_map_record = []
        for layer in self.layers:
            h,attention_map = layer(h)
            attention_map_record.append(attention_map)
        
        h = self.layernorm(h)
        x = self.projection(h)
        return x,attention_map_record

class AutoEncoder(nn.Module):
    '''
    Transformer-based autoencoder model for detection.\n
    Full pipeline:\n
    X (B,L,input_dim)\n
    -> InputEmbedding (B,L,D)\n
    -> PositionalEncoding (B,L,D)\n
    -> TransformerEncoder (B,L,D)\n
    -> TransformerDecoder (B,L,D)
    Parameters:
    ----------
    1. embedding_dimension: dimensions
    2. num_heads: number of MHSA heads
    3. enc_num_layers: transformer layers in encoder
    4. dec_num_layers: transformer layers in decoder
    5. input_size: number of input features
    6. max_length: WINDOW_LENGTH
    7. bottleneck_dim: dim for bottleneck layer
    '''
    def __init__(
            self,
            embedding_dimension:int,
            num_heads:int,
            enc_num_layers:int,
            dec_num_layers:int,
            input_size:int,
            max_length:int,
            bottleneck_dim:int
        ):

        super().__init__()

        self.embed = InputEmbedding(input_size,embedding_dimension)
        self.position_encode = PositionalEncoding(embedding_dimension,max_length)
        
        self.transformer_encoder = Encoder(
            embedding_dimension,
            num_heads,
            input_size,
            max_length,
            enc_num_layers,
            bottleneck_dim
        )
        self.transformer_decoder = Decoder(
            embedding_dimension,
            num_heads,
            dec_num_layers,
            input_size,
            max_length,
            bottleneck_dim
        )
    
    def forward(self,input:torch.Tensor) -> torch.Tensor:
        emb = self.embed(input)
        enc = self.position_encode(emb)
        latent,mu,logvar,encoder_attention = self.transformer_encoder(enc)
        reconstructed_tensor,decoder_attention = self.transformer_decoder(latent)
        return {
            'reconstructed_tensor': reconstructed_tensor,
            'latent': latent,
            'mu': mu,
            'logvar': logvar,
            'encoder_attention': encoder_attention,
            'decoder_attention': decoder_attention,
        }
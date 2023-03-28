import torch 


class PositionwiseFeedForward(torch.nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self, d_in, d_hid, dropout=0.1):
        super().__init__()
        
        ## Initialize two fully connected layers as class attributes, calling them w_1 and w_2
        ## Feed forward
        self.w_1 = torch.nn.Linear(d_in, d_hid) # position-wise
        self.w_2 = torch.nn.Linear(d_hid, d_in) # position-wise
        self.layer_norm = torch.nn.LayerNorm(d_in, eps=1e-6)
        self.dropout = torch.nn.Dropout(dropout)

    def forward(self, x):

        residual = x

        x = self.w_2(torch.nn.functional.relu(self.w_1(x)))
        x = self.dropout(x)
        x += residual

        x = self.layer_norm(x)

        return x


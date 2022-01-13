from torch import nn


class MyAwesomeModel(nn.Module):
    def __init__(self,n_in=784,n_out=10):
        super().__init__()

        self.n_in = n_in
        self.n_out = n_out
    
        self.output = nn.Sequential(
            nn.Linear(self.n_in, 392),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(392 , 196),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(196 ,98),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(98, 49),
            nn.ReLU(),
            nn.Dropout(p=0.2),
            nn.Linear(49 ,self.n_out),
            nn.LogSoftmax(dim=1)
        )
        
    def forward(self, x):
        x = x.view(x.shape[0], -1)       
        return self.output(x)
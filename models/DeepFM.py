import torch
import torch.nn as nn

class DeepFM(nn.Module):
    def __init__(self, input_dims, embedding_dim, mlp_dims, drop_rate=0.1):
        super(DeepFM, self).__init__()
        total_input_dim = int(sum(input_dims)) # n_user + n_movie + n_genre (embedding할 것의 가짓수)

        # FM component
        self.bias = nn.Parameter(torch.zeros((1,)))  # global bias term
        self.fc = nn.Embedding(total_input_dim, 1)  # 1차 bias term (embedding할 것의 가짓수, 1)
        
        self.embedding = nn.Embedding(total_input_dim, embedding_dim)  # embedding할 것의 가짓수, 10 (k=10)
        self.embedding_dim = len(input_dims) * embedding_dim  # 3(field 개수)*10

        # Deep component
        # nn.Linear → ReLU → Dropout → ... → (last) nn.Linear
        mlp_layers = []
        for i, dim in enumerate(mlp_dims):  # mlp_dims=[30, 20, 10]
            if i==0:  # 첫번째 layer인 경우
                mlp_layers.append(nn.Linear(self.embedding_dim, dim))  # 3*10 → 10
            else:
                mlp_layers.append(nn.Linear(mlp_dims[i-1], dim)) # 이전 layer의 차원 → 현재 layer의 차원
            mlp_layers.append(nn.ReLU(True))
            mlp_layers.append(nn.Dropout(drop_rate))
        mlp_layers.append(nn.Linear(mlp_dims[-1], 1))
        self.mlp_layers = nn.Sequential(*mlp_layers)

    def fm(self, x):
        # x : (batch_size, total_num_input)
        embed_x = self.embedding(x)  # shape : (embedding할 것의 개수)*10

        fm_y = self.bias + torch.sum(self.fc(x), dim=1)  # batch_size * embedding할 것의 개수

        square_of_sum = torch.sum(embed_x, dim=1) ** 2  # square_of_sum
        sum_of_square = torch.sum(embed_x ** 2, dim=1)  # sum_of_square
        fm_y += 0.5 * torch.sum(square_of_sum - sum_of_square, dim=1, keepdim=True)
        return fm_y
    
    def mlp(self, x):
        # x : (batch_size, total_num_input)
        embed_x = self.embedding(x)  # shape : (embedding할 것의 개수)*10

        inputs = embed_x.view(-1, self.embedding_dim)  # shape : ?*30
        mlp_y = self.mlp_layers(inputs)  # nn.Linear → ReLU → Dropout → ... → (last) nn.Linear
        return mlp_y

    def forward(self, x):
        # FM component
        fm_y = self.fm(x).squeeze(1)
        
        # Deep component
        mlp_y = self.mlp(x).squeeze(1)
        
        y = torch.sigmoid(fm_y + mlp_y)
        return y
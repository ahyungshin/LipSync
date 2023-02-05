
import torch
import torch.nn as nn
import torch.nn.functional as F

def _xavier_init(net_layer):
    """
    Performs the Xavier weight initialization of the net layer.
    """
    torch.nn.init.xavier_uniform_(net_layer.weight.data)
    

class MLPforNeRF(nn.Module):

    def __init__(self, vp_channels, vd_channels, n_layers = 8, h_channel = 256, res_nfeat = 3) -> None:
        super().__init__()

        self.vp_channels = vp_channels
        self.vd_channels = vd_channels
        self.n_layers = n_layers
        self.h_channel = h_channel

        self.skips = [n_layers // 2]
        self.res_nfeat = res_nfeat
        
        self._make_layers()


    def _make_layers(self):
        dim_aud = 64
        # layers = []
        # layers.append(nn.Conv2d(self.vp_channels, self.h_channel, kernel_size=1, stride= 1, padding=0))
        self.add_module("FeaExt_module_0", nn.Conv2d(self.vp_channels+dim_aud, self.h_channel, kernel_size=1, stride= 1, padding=0))
        # _xavier_init(self._modules["FeaExt_module_0"])
        # self._modules["FeaExt_module_0"].bias.data[:] = 0.0
        
        for i in range(0, self.n_layers - 1):
            if i in self.skips:
                # layers.append(nn.Conv2d(self.h_channel + self.vp_channels, self.h_channel, kernel_size=1, stride=1, padding=0))
                self.add_module("FeaExt_module_%d"%(i + 1), 
                        nn.Conv2d(self.h_channel + self.vp_channels, self.h_channel, kernel_size=1, stride=1, padding=0))
            else:
                # layers.append(nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))
                self.add_module("FeaExt_module_%d"%(i + 1), 
                        nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))

            _xavier_init(self._modules["FeaExt_module_%d"%(i + 1)])
        # self.feature_module_list = layers
        self.add_module("density_module", nn.Conv2d(self.h_channel, 1, kernel_size=1, stride=1, padding=0))
        _xavier_init(self._modules["density_module"])
        self._modules["density_module"].bias.data[:] = 0.0
        
        self.add_module("RGB_layer_0", nn.Conv2d(self.h_channel, self.h_channel, kernel_size=1, stride=1, padding=0))
        _xavier_init(self._modules["RGB_layer_0"])
        
        self.add_module("RGB_layer_1", nn.Conv2d(self.h_channel +  self.vd_channels, self.h_channel//2, kernel_size=1, stride=1, padding=0))
        # _xavier_init(self._modules["RGB_layer_1"])
        # self._modules["RGB_layer_1"].bias.data[:] = 0.0
        
        self.add_module("RGB_layer_2", nn.Conv2d(self.h_channel//2, self.res_nfeat, kernel_size=1, stride=1, padding=0))


    def forward(self, aud, batch_embed_vps, batch_embed_vds):
        '''
        batch_embed_vps: [B, C_1, N_r, N_s]
        batch_embed_vds: [B, C_2, N_r, N_s]
        '''


        x = batch_embed_vps
        aud = aud.unsqueeze(0).unsqueeze(2).unsqueeze(3).repeat(x.shape[0],1,x.shape[2],x.shape[3])
        x = torch.cat([x,aud], dim=1)


        for i in range(self.n_layers):
            x = self._modules["FeaExt_module_%d"%i](x)
            x = F.relu(x)

            if i in self.skips:
                x = torch.cat([batch_embed_vps, x], dim=1)
            
        density = self._modules["density_module"](x)
        x = self._modules["RGB_layer_0"](x)

        x = self._modules["RGB_layer_1"](torch.cat([x, batch_embed_vds], dim = 1))
        x = F.relu(x)

        rgb = self._modules["RGB_layer_2"](x)
        
        density = F.relu(density)
        if self.res_nfeat == 3:
            rgb = torch.sigmoid(rgb)

        return rgb, density



# Audio feature extractor
class AudioAttNet(nn.Module):
    def __init__(self, dim_aud=32, seq_len=8):
        super(AudioAttNet, self).__init__()
        self.seq_len = seq_len
        self.dim_aud = dim_aud
        self.attentionConvNet = nn.Sequential(  # b x subspace_dim x seq_len
            nn.Conv1d(self.dim_aud, 16, kernel_size=3,
                      stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(16, 8, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(8, 4, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(4, 2, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(2, 1, kernel_size=3, stride=1, padding=1, bias=True),
            nn.LeakyReLU(0.02, True)
        )
        self.attentionNet = nn.Sequential(
            nn.Linear(in_features=self.seq_len,
                      out_features=self.seq_len, bias=True),
            nn.Softmax(dim=1)
        )

    def forward(self, x):
        y = x[..., :self.dim_aud].permute(1, 0).unsqueeze(
            0)  # 2 x subspace_dim x seq_len
        y = self.attentionConvNet(y)
        y = self.attentionNet(y.view(1, self.seq_len)).view(self.seq_len, 1)
        # print(y.view(-1).data)
        return torch.sum(y*x, dim=0)
# Model


# Audio feature extractor
class AudioNet(nn.Module):
    def __init__(self, dim_aud=76, win_size=16):
        super(AudioNet, self).__init__()
        self.win_size = win_size
        self.dim_aud = dim_aud
        self.encoder_conv = nn.Sequential(  # n x 29 x 16
            nn.Conv1d(29, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 8
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 32, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 32 x 4
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(32, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 2
            nn.LeakyReLU(0.02, True),
            nn.Conv1d(64, 64, kernel_size=3, stride=2,
                      padding=1, bias=True),  # n x 64 x 1
            nn.LeakyReLU(0.02, True),
        )
        self.encoder_fc1 = nn.Sequential(
            nn.Linear(64, 64),
            nn.LeakyReLU(0.02, True),
            nn.Linear(64, dim_aud),
        )

    def forward(self, x):
        half_w = int(self.win_size/2)
        x = x[:, 8-half_w:8+half_w, :].permute(0, 2, 1)
        x = self.encoder_conv(x).squeeze(-1)
        x = self.encoder_fc1(x).squeeze()
        return x
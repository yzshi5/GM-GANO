import torch
import numpy as np
import pylab as plt
import torch.nn.functional as F
import torch.nn as nn

def kernel_loc(in_chan=2, up_dim=32):
    """
        Kernel network apply on grid
    """
    layers = nn.Sequential(
                nn.Linear(in_chan, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, up_dim, bias=True), torch.nn.GELU(),
                nn.Linear(up_dim, 1, bias=False)
            )
    return layers

class SpectralConv1d(nn.Module):
    """
    1D Fourier layer. It does FFT, linear transform, and Inverse FFT.  
    
    key_parameters:
    ---------------
    dim1 : output dimension 
    modes1 :  <= (dim1//2) & <= (x.shape[-1] // 2)
    """    
    def __init__(self, in_channels, out_channels, dim1, modes1):
        super(SpectralConv1d, self).__init__()

        in_channels = int(in_channels)
        out_channels = int(out_channels)

        self.in_channels = in_channels
        self.out_channels = out_channels
        #at most floor(N/2) + 1, by default, modes1 = dim1//2 
        self.modes1 = modes1  
        self.dim1 = dim1
        
        self.scale = (1 / (in_channels*out_channels))
        self.weights1 = nn.Parameter(self.scale * torch.randn(in_channels, out_channels, self.modes1, dtype=torch.cfloat))

    # Complex multiplication
    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x ), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

    def forward(self, x, dim1=None):
        if dim1 is not None:
            self.dim1 = dim1

        batchsize = x.shape[0]
        #Compute Fourier coeffcients up to factor of e^(- something constant)
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels, self.dim1//2 + 1,  device=x.device, dtype=torch.cfloat)
        out_ft[:, :, :self.modes1] = self.compl_mul1d(x_ft[:, :, :self.modes1], self.weights1)

        #Return to physical space
        x = torch.fft.irfft(out_ft, n=dim1)
        return x


class pointwise_op(nn.Module):
    def __init__(self, in_channel, out_channel,dim1):
        super(pointwise_op,self).__init__()
        self.conv = nn.Conv1d(int(in_channel), int(out_channel), 1)
        self.dim1 = int(dim1)

    def forward(self,x, dim1 = None):
        if dim1 is None:
            dim1 = self.dim1
            
        x_out = self.conv(x)
        x_out = torch.nn.functional.interpolate(x_out, size = dim1,mode = 'linear',align_corners=True)
        return x_out
    
    
class Generator(nn.Module):
    def __init__(self, in_width, width,  ndim,pad = 0, factor = 3/4, training=False):
        super(Generator, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv.
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        Parameters: 
        ------------
        in_width : int
            input_channel number
        width : int
            number of hidden FNO layers
        training : bool
            True : no truncation
            False : for evaluation, with truncation 6400 -> 6000 
        """
        self.in_width = in_width # input channel
        self.width = width 
        self.ndim = ndim
        self.factor = factor
        self.padding = pad  # pad the domain if input is non-periodic
        self.training = training
        
        self.final_dim = self.ndim + self.padding

        self.fc0 = nn.Linear(self.in_width, self.width)

        # input: 6400, output D1*factor = 4800, model=2400
        #self.conv0 = SpectralConv1d(self.width, 2*factor*self.width, 4800, 2400)  # last mode < D1 * (factor) / 2 
        self.conv0 = SpectralConv1d(self.width, 2*factor*self.width, int(self.factor*self.final_dim), int(self.factor*self.final_dim)//2)  # last mode < D1 * (factor) / 2 

        # input: 4800, output D1//2=3200, mode=1200
        # self.conv1 = SpectralConv1d(2*factor*self.width, 4*factor*self.width, 3200, 1600)
        self.conv1 = SpectralConv1d(2*factor*self.width, 4*factor*self.width, self.final_dim//2, self.final_dim//4)

        # input: 3200, output D1//4=1600, mode=600
        # self.conv2 = SpectralConv1d(4*factor*self.width, 8*factor*self.width, 1600,800) # half about conv1
        self.conv2 = SpectralConv1d(4*factor*self.width, 8*factor*self.width, self.final_dim//4, self.final_dim//8) # half about conv1

        # input: 1600, output D1//8=800, mode=400
        # self.conv2_1 = SpectralConv1d(8*factor*self.width, 16*factor*self.width,800,400)
        self.conv2_1 = SpectralConv1d(8*factor*self.width, 16*factor*self.width,self.final_dim//8, self.final_dim//16)

        # input: 800, output D1//4=1600, mode=400
        # self.conv2_9 = SpectralConv1d(16*factor*self.width, 8*factor*self.width, 1600,400)
        self.conv2_9 = SpectralConv1d(16*factor*self.width, 8*factor*self.width, self.final_dim//4,self.final_dim//16)

        # input 1600: output D1//2=3200, mode=800
        # self.conv3 = SpectralConv1d(16*factor*self.width, 4*factor*self.width, 3200, 800)
        self.conv3 = SpectralConv1d(16*factor*self.width, 4*factor*self.width, self.final_dim//2, self.final_dim//8)

        # input 3200, output 4800, mode=1600
        # self.conv4 = SpectralConv1d(8*factor*self.width, 2*factor*self.width, 4800, 1600)
        self.conv4 = SpectralConv1d(8*factor*self.width, 2*factor*self.width, int(self.factor*self.final_dim), self.final_dim//4)

        # input 4800,  output 6400,  model=2400
        # self.conv5 = SpectralConv1d(4*factor*self.width, self.width, 6400,2400) # will be reshaped
        self.conv5 = SpectralConv1d(4*factor*self.width, self.width, self.final_dim, int(self.factor*self.final_dim)//2) # will be reshaped

        self.w0 = pointwise_op(self.width,2*factor*self.width, int(self.factor*self.final_dim)) #inconsistence with dim
        
        self.w1 = pointwise_op(2*factor*self.width, 4*factor*self.width, self.final_dim//2) #
        
        self.w2 = pointwise_op(4*factor*self.width, 8*factor*self.width, self.final_dim//4) #
        
        self.w2_1 = pointwise_op(8*factor*self.width, 16*factor*self.width, self.final_dim//8)
        
        self.w2_9 = pointwise_op(16*factor*self.width, 8*factor*self.width, self.final_dim//4)
        
        self.w3 = pointwise_op(16*factor*self.width, 4*factor*self.width, self.final_dim//2) #
        
        self.w4 = pointwise_op(8*factor*self.width, 2*factor*self.width, int(self.factor*self.final_dim))
        
        self.w5 = pointwise_op(4*factor*self.width, self.width, self.final_dim) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        # first three are normalized 3C waveforms, last three are associated PGAs
        self.fc2 = nn.Linear(4*self.width, 6)
        
    def forward(self, x, label):
        
        label = label.repeat(1, (self.ndim + self.padding), 1) 
        
        x = torch.cat([x, label], axis=2)
        
        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)   
        x_fc0 = x_fc0.permute(0, 2, 1)
        #print("G x_fc0:",x_fc0.shape)

        # D1 = 6400
        D1 = x_fc0.shape[-1]
        
        x1_c0 = self.conv0(x_fc0,int(D1*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print("G x_c0:",x_c0.shape)
        
        x1_c1 = self.conv1(x_c0, D1//2)
        x2_c1 = self.w1(x_c0 ,D1//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print("G x_c1:",x_c1.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4)
        x2_c2 = self.w2(x_c1 ,D1//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print("G x_c2:",x_c2.shape)
        
        x1_c2_1 = self.conv2_1(x_c2,D1//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)
        #print("G x_c2_1:",x_c2_1.shape)
        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 
        #print("G x_c2_9:",x_c2_9.shape)
        
        x1_c3 = self.conv3(x_c2_9,D1//2)
        x2_c3 = self.w3(x_c2_9,D1//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)
        #print("G x_c3:",x_c3.shape)
        
        x1_c4 = self.conv4(x_c3,int(D1*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)
        #print("G x_c4:",x_c4.shape)
        
        x1_c5 = self.conv5(x_c4,D1)
        x2_c5 = self.w5(x_c4,D1)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        #print("G x_c5:",x_c5.shape)
        
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        
        """
        if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding]        
        """
        
        x = x_c5.permute(0, 2, 1)
        #print("x shape", x.shape)
        
        x = self.fc1(x)
        x = F.gelu(x)
        
        # be sure to normalize the inpout conditional varaibles to [-1, 1], rather than [0, 1]
        x_out = self.fc2(x).permute(0, 2, 1)
        x_out[:,3:] = torch.mean(x_out[:,3:], dim=-1).unsqueeze(2)
        x_out = torch.tanh(x_out)  

        if not self.training:
            x_out = x_out[:,:,:self.ndim]  
        return x_out


class Discriminator(nn.Module):
    def __init__(self, in_width, width,  ndim, kernel_dim=16,pad = 0, factor = 3/4):
        super(Discriminator, self).__init__()

        """
        The overall network. It contains 4 layers of the Fourier layer.
        1. Lift the input to the desire channel dimension by self.fc0 .
        2. 4 layers of the integral operators u' = (W + K)(u).
            W defined by self.w; K defined by self.conv .
        3. Project from the channel space to the output space by self.fc1 and self.fc2 .
        
        """
        self.in_width = in_width # input channel
        self.width = width 
        self.factor = factor
        self.ndim=ndim
        self.padding = pad  # pad the domain if input is non-periodic
        self.kernel_dim=kernel_dim
        self.final_dim = self.ndim + self.padding
        
        self.fc0 = nn.Linear(self.in_width, self.width)

        # input: 6400, output D1*factor = 4800, model=2400
        self.conv0 = SpectralConv1d(self.width, 2*factor*self.width, int(self.factor*self.final_dim), int(self.factor*self.final_dim)//2)  # last mode < D1 * (factor) / 2 

        # input: 4800, output D1//2=3200, mode=1200
        # self.conv1 = SpectralConv1d(2*factor*self.width, 4*factor*self.width, 3200, 1600)
        self.conv1 = SpectralConv1d(2*factor*self.width, 4*factor*self.width, self.final_dim//2, self.final_dim//4)

        # input: 3200, output D1//4=1600, mode=600
        # self.conv2 = SpectralConv1d(4*factor*self.width, 8*factor*self.width, 1600,800) # half about conv1
        self.conv2 = SpectralConv1d(4*factor*self.width, 8*factor*self.width, self.final_dim//4, self.final_dim//8) # half about conv1

        # input: 1600, output D1//8=800, mode=400
        # self.conv2_1 = SpectralConv1d(8*factor*self.width, 16*factor*self.width,800,400)
        self.conv2_1 = SpectralConv1d(8*factor*self.width, 16*factor*self.width,self.final_dim//8, self.final_dim//16)

        # input: 800, output D1//4=1600, mode=400
        # self.conv2_9 = SpectralConv1d(16*factor*self.width, 8*factor*self.width, 1600,400)
        self.conv2_9 = SpectralConv1d(16*factor*self.width, 8*factor*self.width, self.final_dim//4,self.final_dim//16)

        # input 1600: output D1//2=3200, mode=800
        # self.conv3 = SpectralConv1d(16*factor*self.width, 4*factor*self.width, 3200, 800)
        self.conv3 = SpectralConv1d(16*factor*self.width, 4*factor*self.width, self.final_dim//2, self.final_dim//8)

        # input 3200, output 4800, mode=1600
        # self.conv4 = SpectralConv1d(8*factor*self.width, 2*factor*self.width, 4800, 1600)
        self.conv4 = SpectralConv1d(8*factor*self.width, 2*factor*self.width, int(self.factor*self.final_dim), self.final_dim//4)

        # input 4800,  output 6400,  model=2400
        # self.conv5 = SpectralConv1d(4*factor*self.width, self.width, 6400,2400) # will be reshaped
        self.conv5 = SpectralConv1d(4*factor*self.width, self.width, self.final_dim, int(self.factor*self.final_dim)//2) # will be reshaped

        self.w0 = pointwise_op(self.width,2*factor*self.width, int(self.factor*self.final_dim)) #inconsistence with dim
        
        self.w1 = pointwise_op(2*factor*self.width, 4*factor*self.width, self.final_dim//2) #
        
        self.w2 = pointwise_op(4*factor*self.width, 8*factor*self.width, self.final_dim//4) #
        
        self.w2_1 = pointwise_op(8*factor*self.width, 16*factor*self.width, self.final_dim//8)
        
        self.w2_9 = pointwise_op(16*factor*self.width, 8*factor*self.width, self.final_dim//4)
        
        self.w3 = pointwise_op(16*factor*self.width, 4*factor*self.width, self.final_dim//2) #
        
        self.w4 = pointwise_op(8*factor*self.width, 2*factor*self.width, int(self.factor*self.final_dim))
        
        self.w5 = pointwise_op(4*factor*self.width, self.width, self.final_dim) # will be reshaped

        self.fc1 = nn.Linear(2*self.width, 4*self.width)
        self.fc2 = nn.Linear(4*self.width, 1)

        self.knet = kernel_loc(1, self.kernel_dim)

    def forward(self, x, label):
       
        # x shape is [batch_size, 4 , 12000]
        # label shape is [batch_size, 1, 4]
        label = label.repeat(1, self.ndim+self.padding, 1) 

        x = x.permute(0, 2, 1)        
        x = torch.cat([x, label], dim=2)

        grid = self.get_grid(x.shape, x.device)

        #print("D x shape: {}".format(x.shape))
        x_fc0 = self.fc0(x)
        x_fc0 = F.gelu(x_fc0)   
        x_fc0 = x_fc0.permute(0, 2, 1)

        D1 = x_fc0.shape[-1]
        
        x1_c0 = self.conv0(x_fc0,int(D1*self.factor))
        x2_c0 = self.w0(x_fc0,int(D1*self.factor))
        x_c0 = x1_c0 + x2_c0
        x_c0 = F.gelu(x_c0)
        #print(x.shape)

        x1_c1 = self.conv1(x_c0 ,D1//2)
        x2_c1 = self.w1(x_c0 ,D1//2)
        x_c1 = x1_c1 + x2_c1
        x_c1 = F.gelu(x_c1)
        #print(x.shape)

        x1_c2 = self.conv2(x_c1 ,D1//4)
        x2_c2 = self.w2(x_c1 ,D1//4)
        x_c2 = x1_c2 + x2_c2
        x_c2 = F.gelu(x_c2 )
        #print(x.shape)
        
        x1_c2_1 = self.conv2_1(x_c2,D1//8)
        x2_c2_1 = self.w2_1(x_c2,D1//8)
        x_c2_1 = x1_c2_1 + x2_c2_1
        x_c2_1 = F.gelu(x_c2_1)
        
        x1_c2_9 = self.conv2_9(x_c2_1,D1//4)
        x2_c2_9 = self.w2_9(x_c2_1,D1//4)
        x_c2_9 = x1_c2_9 + x2_c2_9
        x_c2_9 = F.gelu(x_c2_9)
        x_c2_9 = torch.cat([x_c2_9, x_c2], dim=1) 

        x1_c3 = self.conv3(x_c2_9,D1//2)
        x2_c3 = self.w3(x_c2_9,D1//2)
        x_c3 = x1_c3 + x2_c3
        x_c3 = F.gelu(x_c3)
        x_c3 = torch.cat([x_c3, x_c1], dim=1)

        x1_c4 = self.conv4(x_c3,int(D1*self.factor))
        x2_c4 = self.w4(x_c3,int(D1*self.factor))
        x_c4 = x1_c4 + x2_c4
        x_c4 = F.gelu(x_c4)
        x_c4 = torch.cat([x_c4, x_c0], dim=1)

        x1_c5 = self.conv5(x_c4,D1)
        x2_c5 = self.w5(x_c4,D1)
        x_c5 = x1_c5 + x2_c5
        x_c5 = F.gelu(x_c5)
        
        x_c5 = torch.cat([x_c5, x_fc0], dim=1)
        
        """
         if self.padding!=0:
            x_c5 = x_c5[..., :-self.padding]       
        """

        # x shape [n, 4000,  64]
        x = x_c5.permute(0, 2, 1)

        x = self.fc1(x)
        x = F.gelu(x)
        x = self.fc2(x)

 
        """
        ## instead of using the mean as the functional,  we can choose a kernel for last functional operation. 
        ## example can be found in https://github.com/neuraloperator/GANO/blob/main/GANO_volcano.ipynb  
        # comment 'x= torch.mean(x)' if you use a kernel for last function operator 
        print("ok")
        kx = self.knet(grid)
        x = torch.einsum('bik,bik->bk', kx, x)/(self.ndim+self.padding)
        """

        # use mean for the last step
        x = torch.mean(x)

        return x
    
    def get_grid(self, shape, device):
        batchsize, size_x = shape[0], shape[1]
        gridx = torch.tensor(np.linspace(0, 1, size_x), dtype=torch.float)
        gridx = gridx.reshape(1, size_x, 1).repeat([batchsize, 1, 1])
        return gridx.to(device)
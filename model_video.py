import torch.nn.functional as F
from torch import nn
from torch.nn import Module
import torch
from audioUtils.hparams import hparams

class MyUpsample(Module):
    __constants__ = ['size', 'scale_factor', 'mode', 'align_corners', 'name']

    def __init__(self, size=None, scale_factor=None, mode='nearest', align_corners=None):
        super(MyUpsample, self).__init__()
        self.name = type(self).__name__
        self.size = size
        self.scale_factor = scale_factor if scale_factor else None
        self.mode = mode
        self.align_corners = align_corners

    def forward(self, input):
        return F.interpolate(input, self.size, self.scale_factor, self.mode, self.align_corners)

    def extra_repr(self):
        if self.scale_factor is not None:
            info = 'scale_factor=' + str(self.scale_factor)
        else:
            info = 'size=' + str(self.size)
        info += ', mode=' + self.mode
        return info


class VideoGenerator(nn.Module):
    # initializers
    def __init__(self, d=128, dim_neck=32, use_window=True, use_256=False):
        super(VideoGenerator, self).__init__()
        self.deconv1 = nn.ConvTranspose2d(256, d*8, 4, 1, 0)
        self.deconv1_bn = nn.BatchNorm2d(d*8)
        self.deconv2 = nn.ConvTranspose2d(d*8, d*4, 4, 2, 1)
        self.deconv2_bn = nn.BatchNorm2d(d*4)
        self.deconv3 = nn.ConvTranspose2d(d*4, d*2, 4, 2, 1)
        self.deconv3_bn = nn.BatchNorm2d(d*2)
        self.deconv4 = nn.ConvTranspose2d(d*2, d, 4, 2, 1)
        self.deconv4_bn = nn.BatchNorm2d(d)
        self.deconv5 = nn.ConvTranspose2d(d, d//2, 4, 2, 1)
        self.deconv5_bn = nn.BatchNorm2d(d//2)
        if use_256:
            self.deconv6 = nn.ConvTranspose2d(d // 2, d // 4, 4, 2, 1)
            self.deconv6_bn = nn.BatchNorm2d(d // 4)
            self.deconv7 = nn.ConvTranspose2d(d // 4, 3, 4, 2, 1)
        else:
            self.deconv7 = nn.ConvTranspose2d(d // 2, 3, 4, 2, 1)
        if not use_window:
            self.lstm = nn.LSTM(dim_neck*2, 256, 1, batch_first=True)
        else:
            self.window = nn.Conv1d(in_channels=dim_neck*2, out_channels=256, kernel_size=64, stride=4, padding=30)
        self.use_window = use_window
        self.use_256 = use_256

    # weight_init
    def weight_init(self, mean, std):
        for m in self._modules:
            normal_init(self._modules[m], mean, std)

    # forward method
    def forward(self, input, return_feature=False):
        # x = F.relu(self.deconv1(input))
        # print(input.shape)
        if self.use_window:
            input = self.window(input.transpose(1,2)).transpose(1,2)
        else:
            input, _ = self.lstm(input)
        # print(input.shape)
        batch_sz, num_frames, feat_dim = input.shape
        input = input.reshape(-1, feat_dim, 1, 1)
        x = F.relu(self.deconv1_bn(self.deconv1(input)))
        x = F.relu(self.deconv2_bn(self.deconv2(x)))
        x = F.relu(self.deconv3_bn(self.deconv3(x)))
        x = F.relu(self.deconv4_bn(self.deconv4(x)))
        x = F.relu(self.deconv5_bn(self.deconv5(x)))
        if self.use_256:
            x = F.relu(self.deconv6_bn(self.deconv6(x)))
        x = torch.tanh(self.deconv7(x))
        x = x.reshape(batch_sz, num_frames, x.shape[1], x.shape[2], x.shape[3])
        if return_feature:
            return x, input
        return x


def normal_init(m, mean, std):
    if isinstance(m, nn.ConvTranspose2d) or isinstance(m, nn.Conv2d):
        m.weight.data.normal_(mean, std)
        m.bias.data.zero_()
        

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def conv3d(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

# Upsale the spatial size by a factor of 2
def upBlock(in_planes, out_planes):
    block = nn.Sequential(
        # nn.Upsample(scale_factor=2, mode='nearest'),
        # conv3x3(in_planes, out_planes),
        MyUpsample(scale_factor=(1,2,2), mode='nearest'),
        conv3d(in_planes, out_planes),
        nn.BatchNorm3d(out_planes),
        nn.ReLU(True))
    return block

class ResBlock(nn.Module):
    def __init__(self, channel_num):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num),
            nn.ReLU(True),
            conv3x3(channel_num, channel_num),
            nn.BatchNorm2d(channel_num))
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x
        out = self.block(x)
        out += residual
        out = self.relu(out)
        return out

class STAGE2_G(nn.Module):
    def __init__(self, residual=False):
        super(STAGE2_G, self).__init__()
        self.STAGE1_G = VideoGenerator()
        # fix parameters of stageI GAN
#         for param in self.STAGE1_G.parameters():
#             param.requires_grad = False
        self.define_module()
        self.residual_video = residual

    def _make_layer(self, block, channel_num):
        layers = []
        for i in range(4):
            layers.append(block(channel_num))
        return nn.Sequential(*layers)

    def define_module(self):
        ngf = 32
        # TEXT.DIMENSION -> GAN.CONDITION_DIM
        # --> 4ngf x 32 x 32
        self.encoder = nn.Sequential(
            conv3x3(3, ngf),
            nn.ReLU(True),
            nn.Conv2d(ngf, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            nn.Conv2d(ngf * 2, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.hr_joint = nn.Sequential(
            conv3x3(256 + ngf * 4, ngf * 4),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True))
        self.residual = self._make_layer(ResBlock, ngf * 4)
        # --> 2ngf x 64 x 64
        self.upsample1 = upBlock(ngf * 4, ngf * 2)
        # --> ngf x 128 x 128
        self.upsample2 = upBlock(ngf * 2, ngf)
        # --> ngf // 2 x 256 x 256
        self.upsample3 = upBlock(ngf, ngf // 2)
        # --> ngf // 4 x 512 x 512
        self.upsample4 = upBlock(ngf // 2, ngf // 4)
        # --> 3 x 512 x 512
        self.img = nn.Sequential(
            conv3d(ngf // 4, 3),
            nn.Tanh())

    def forward(self, input, train=False):
        stage1_video, audio_embedding = self.STAGE1_G(input, return_feature=True)
        batch_sz, num_frames, _,_,_ = stage1_video.shape
        encoded_frames = self.encoder(stage1_video.reshape(batch_sz*num_frames,3,128,128))

        c_code = audio_embedding.reshape(batch_sz*num_frames,256,1,1)
        c_code = c_code.repeat(1, 1, 32, 32)
        i_c_code = torch.cat([encoded_frames, c_code], 1)
        h_code = self.hr_joint(i_c_code)
        h_code = self.residual(h_code) # (bs*num_frame)*4ngf*32*32

        h_code = h_code.reshape(batch_sz, num_frames, -1, 32, 32).transpose(2,1)
        h_code = self.upsample1(h_code)
        h_code = self.upsample2(h_code)
        h_code = self.upsample3(h_code)
        h_code = self.upsample4(h_code)

        stage2_video = self.img(h_code)
        stage2_video = stage2_video.transpose(2,1).reshape(batch_sz, num_frames, 3, 512, 512)

        if self.residual_video:
            stage2_video = MyUpsample(scale_factor=(1,4,4), mode='nearest')(stage1_video) + stage2_video

        if train:
            return stage1_video, stage2_video
        return stage2_video


class ConvLayer(nn.Module):
    
    def __init__(self, in_channels=1, out_channels=256):
        '''Constructs the ConvLayer with a specified input and output size.
           param in_channels: input depth of an image, default value = 1
           param out_channels: output depth of the convolutional layer, default value = 256
           '''
        super(ConvLayer, self).__init__()

        # defining a convolutional layer of the specified size
        self.conv = nn.Conv2d(in_channels, out_channels, 
                              kernel_size=9, stride=1, padding=0)

    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input to the layer; an input image
           return: a relu-activated, convolutional layer
           '''
        # applying a ReLu activation to the outputs of the conv layer
        features = F.relu(self.conv(x)) # will have dimensions (batch_size, 20, 20, 256)
        return features

class PrimaryCaps(nn.Module):
    
    def __init__(self, num_capsules=8, in_channels=256, out_channels=32):
        '''Constructs a list of convolutional layers to be used in 
           creating capsule output vectors.
           param num_capsules: number of capsules to create
           param in_channels: input depth of features, default value = 256
           param out_channels: output depth of the convolutional layers, default value = 32
           '''
        super(PrimaryCaps, self).__init__()

        # creating a list of convolutional layers for each capsule I want to create
        # all capsules have a conv layer with the same parameters
        self.capsules = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, 
                      kernel_size=9, stride=2, padding=0)
            for _ in range(num_capsules)])
    
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; features from a convolutional layer
           return: a set of normalized, capsule output vectors
           '''
        # get batch size of inputs
        batch_size = x.size(0)
        # reshape convolutional layer outputs to be (batch_size, vector_dim=1152, 1)
        u = [capsule(x).view(batch_size, 32 * 6 * 6, 1) for capsule in self.capsules]
        # stack up output vectors, u, one for each capsule
        u = torch.cat(u, dim=-1)
        # squashing the stack of vectors
        u_squash = self.squash(u)
        return u_squash
    def squash(self, input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
           '''
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor    

# to get transpose softmax function

# dynamic routing
def dynamic_routing(b_ij, u_hat, squash, routing_iterations=3):
    '''Performs dynamic routing between two capsule layers.
       param b_ij: initial log probabilities that capsule i should be coupled to capsule j
       param u_hat: input, weighted capsule vectors, W u
       param squash: given, normalizing squash function
       param routing_iterations: number of times to update coupling coefficients
       return: v_j, output capsule vectors
       '''    
    # update b_ij, c_ij for number of routing iterations
    for iteration in range(routing_iterations):
        # softmax calculation of coupling coefficients, c_ij
        c_ij = softmax(b_ij, dim=2)

        # calculating total capsule inputs, s_j = sum(c_ij*u_hat)
        s_j = (c_ij * u_hat).sum(dim=2, keepdim=True)

        # squashing to get a normalized vector output, v_j
        v_j = squash(s_j)

        # if not on the last iteration, calculate agreement and new b_ij
        if iteration < routing_iterations - 1:
            # agreement
            a_ij = (u_hat * v_j).sum(dim=-1, keepdim=True)
            
            # new b_ij
            b_ij = b_ij + a_ij
    
    return v_j # return latest v_j

class DigitCaps(nn.Module):
    
    def __init__(self, num_capsules=10, previous_layer_nodes=32*6*6, 
                 in_channels=8, out_channels=16):
        '''Constructs an initial weight matrix, W, and sets class variables.
           param num_capsules: number of capsules to create
           param previous_layer_nodes: dimension of input capsule vector, default value = 1152
           param in_channels: number of capsules in previous layer, default value = 8
           param out_channels: dimensions of output capsule vector, default value = 16
           '''
        super(DigitCaps, self).__init__()

        # setting class variables
        self.num_capsules = num_capsules
        self.previous_layer_nodes = previous_layer_nodes # vector input (dim=1152)
        self.in_channels = in_channels # previous layer's number of capsules

        # starting out with a randomly initialized weight matrix, W
        # these will be the weights connecting the PrimaryCaps and DigitCaps layers
        self.W = nn.Parameter(torch.randn(num_capsules, previous_layer_nodes, 
                                          in_channels, out_channels))

    def forward(self, u):
        '''Defines the feedforward behavior.
           param u: the input; vectors from the previous PrimaryCaps layer
           return: a set of normalized, capsule output vectors
           '''
        
        # adding batch_size dims and stacking all u vectors
        u = u[None, :, :, None, :]
        # 4D weight matrix
        W = self.W[:, None, :, :, :]
        
        # calculating u_hat = W*u
        u_hat = torch.matmul(u, W)
        # getting the correct size of b_ij
        # setting them all to 0, initially
        b_ij = torch.zeros(*u_hat.size())
        
        # moving b_ij to GPU, if available
        if TRAIN_ON_GPU:
            b_ij = b_ij.cuda()

        # update coupling coefficients and calculate v_j
        v_j = dynamic_routing(b_ij, u_hat, self.squash, routing_iterations=3)

        return v_j # return final vector outputs

    def squash(self, input_tensor):
        '''Squashes an input Tensor so it has a magnitude between 0-1.
           param input_tensor: a stack of capsule inputs, s_j
           return: a stack of normalized, capsule output vectors, v_j
           '''
        # same squash function as before
        squared_norm = (input_tensor ** 2).sum(dim=-1, keepdim=True)
        scale = squared_norm / (1 + squared_norm) # normalization coeff
        output_tensor = scale * input_tensor / torch.sqrt(squared_norm)    
        return output_tensor

class Decodar(nn.Module):
    
    def __init__(self, input_vector_length=16, input_capsules=10, hidden_dim=512):
        '''Constructs an series of linear layers + activations.
           param input_vector_length: dimension of input capsule vector, default value = 16
           param input_capsules: number of capsules in previous layer, default value = 10
           param hidden_dim: dimensions of hidden layers, default value = 512
           '''
        super(Decodar, self).__init__()
        
        # calculate input_dim
        input_dim = input_vector_length * input_capsules
        
        # define linear layers + activations
        self.linear_layers = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), # first hidden layer
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim*2), # second, twice as deep
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim*2, 28*28), # can be reshaped into 28*28 image
            nn.Sigmoid() # sigmoid activation to get output pixel values in a range from 0-1
            )
        
    def forward(self, x):
        '''Defines the feedforward behavior.
           param x: the input; vectors from the previous DigitCaps layer
           return: two things, reconstructed images and the class scores, y
           '''
        classes = (x ** 2).sum(dim=-1) ** 0.5
        classes = F.softmax(classes, dim=-1)
        
        # find the capsule with the maximum vector length
        # here, vector length indicates the probability of a class' existence
        _, max_length_indices = classes.max(dim=1)
        
        # create a sparse class matrix
        sparse_matrix = torch.eye(10) # 10 is the number of classes
        if TRAIN_ON_GPU:
            sparse_matrix = sparse_matrix.cuda()
        # get the class scores from the "correct" capsule
        y = sparse_matrix.index_select(dim=0, index=max_length_indices.data)
        
        # create reconstructed pixels
        x = x * y[:, :, None]
        # flatten image into a vector shape (batch_size, vector_dim)
        flattened_x = x.contiguous().view(x.size(0), -1)
        # create reconstructed image vectors
        reconstructions = self.linear_layers(flattened_x)
        
        # return reconstructions and the class scores, y
        return reconstructions, y

class VideoEncoder(nn.Module):
    
    def __init__(self):
        '''Constructs a complete Capsule Network.'''
        super(CapsuleNetwork, self).__init__()
        self.conv_layer = ConvLayer()
        self.primary_capsules = PrimaryCaps()
        self.digit_capsules = DigitCaps()
        self.decodar = Decodar()
                
    def forward(self, images):
        '''Defines the feedforward behavior.
           param images: the original MNIST image input data
           return: output of DigitCaps layer, reconstructed images, class scores
           '''
        primary_caps_output = self.primary_capsules(self.conv_layer(images))
        caps_output = self.digit_capsules(primary_caps_output).squeeze().transpose(0,1)
        reconstructions, y = self.decodar(caps_output)
        return caps_output, reconstructions, y


#class VideoEncoder(nn.Module):
 #   def __init__(self):
  #      super(VideoEncoder, self).__init__()
   #     self.encoder = nn.Sequential(
    #        nn.Conv3d(3, 64, kernel_size=(3,4,4), stride=(1,2,2), padding=1, bias=False), # 32*256*256
     #       nn.BatchNorm3d(64),
      #      nn.ReLU(True),
       #     nn.Conv3d(64, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False), # 32*128*128
        #    nn.BatchNorm3d(128),
         #   nn.ReLU(True),
            #nn.Conv3d(128, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False), # 32*64*64
            #nn.BatchNorm3d(256),
            #nn.ReLU(True),
            #nn.Conv3d(256, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*32*32
            #nn.BatchNorm3d(256),
            #nn.ReLU(True),
            #nn.Conv3d(256, 256, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*16*16
            #nn.BatchNorm3d(256),
            #nn.ReLU(True),
            #nn.Conv3d(256, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*8*8
            #nn.BatchNorm3d(128),
            #nn.ReLU(True),
            #nn.Conv3d(128, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*4*4
            #nn.BatchNorm3d(128),
            #nn.ReLU(True),
            #nn.Conv3d(128, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*2*2
            #nn.BatchNorm3d(128),
           # nn.ReLU(True),
            #nn.Conv3d(128, 128, kernel_size=(3, 4, 4), stride=(1, 2, 2), padding=1, bias=False),  # 32*1*1
            #nn.BatchNorm3d(128),
            #nn.ReLU(True),
       # )
        #self.projection = nn.Sequential(
#            nn.Conv1d(128, 64, kernel_size=4, stride=2, padding=1, bias=False),
 #           nn.BatchNorm1d(64),
#            nn.ReLU(True),
 #           nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
  #          nn.BatchNorm1d(64),
   #         nn.ReLU(True),
   #         nn.Conv1d(64, 64, kernel_size=4, stride=2, padding=1, bias=False),
   #         nn.BatchNorm1d(64),
   #         nn.ReLU(True),
   #         nn.Conv1d(64, 64, kernel_size=3, stride=1, padding=1, bias=True)
   #     )

   # def forward(self, x):
        # batch * time * channel * 512 * 512
    #    batch_sz, num_frames, _, _, _ = x.shape
    #    x = x.transpose(2, 1)
    #    x = self.encoder(x) # batch * 128 * time * 1 * 1
    #    x = x.reshape(batch_sz, 128, num_frames)
     #   x = self.projection(x)
        # print(x.shape)
     #   return x.transpose(1,2)


import torch
import torch.nn as nn
import torchvision

class DepthwiseSeperable(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1):
        super(DepthwiseSeperable, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        self.pointwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=self.stride
        )
        
        self.depthwise = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            kernel_size=self.kernel_size,
            stride=self.stride,
            padding=self.padding,
            groups=self.in_channels
        )
        
    def forward(self, x):
        x = self.pointwise(x)
        x = self.depthwise(x)
        return x
        

class BottleNeck(nn.Module):
    def __init__(self, in_chan, out_chan, reduction_ratio=4, rescaled_identity=False):
        super(BottleNeck, self).__init__()
        self.in_channels = in_chan
        self.out_channels = out_chan
        self.rescaled_identity = rescaled_identity
        self.down_channels = self.out_channels//reduction_ratio
        
        self.conv1x1_down = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.down_channels,
            kernel_size=1,
            stride=1
        )
        
        self.Lite = nn.Sequential(
            nn.Conv2d(
                in_channels=self.down_channels,
                out_channels=self.down_channels,
                kernel_size=1,
                stride=1
            ),
            DepthwiseSeperable(
                in_channels=self.down_channels,
                out_channels=self.down_channels,
                kernel_size=3
            ),
            nn.BatchNorm2d(self.down_channels),
            nn.ReLU()
        )
    
        self.conv1x1_up = nn.Conv2d(
            in_channels=self.down_channels,
            out_channels=self.out_channels,
            kernel_size=1,
            stride=1
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.ag_gate = nn.Sequential(
            nn.Linear(self.down_channels, self.down_channels//16),
            nn.ReLU(),
            nn.Linear(self.down_channels//16, self.down_channels),
            nn.Sigmoid()
        )
        
        self.bn = nn.BatchNorm2d(self.out_channels)
        self.relu = nn.ReLU()
        
        self.conv_identity = nn.Conv2d(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            kernel_size=1
        )
        
    def forward(self, x):
        identity = x
        x = self.conv1x1_down(x)
        
        # branch 1xLite
        lite1x = self.Lite(x)
        
        # branch 2xLite
        lite2x = self.Lite(x)
        lite2x = self.Lite(lite2x)
        
        # branch 3xLite
        lite3x = self.Lite(x)
        for _ in range(2):
            lite3x = self.Lite(lite3x)
            
        # branch 4xLite
        lite4x = self.Lite(x)
        for _ in range(3):
            lite4x = self.Lite(lite4x)
        
        # ag_gate 1xLite
        ag_gate_1x = self.gap(lite1x)
        ag_gate_1x = ag_gate_1x.view(ag_gate_1x.size(0), -1)
        ag_gate_1x = self.ag_gate(ag_gate_1x).view(ag_gate_1x.size(0), -1, 1, 1)
        
        # ag_gate 2xLite
        ag_gate_2x = self.gap(lite2x)
        ag_gate_2x = ag_gate_2x.view(ag_gate_2x.size(0), -1)
        ag_gate_2x = self.ag_gate(ag_gate_2x).view(ag_gate_2x.size(0), -1, 1, 1)
        
        # ag_gate 3xLite
        ag_gate_3x = self.gap(lite3x)
        ag_gate_3x = ag_gate_3x.view(ag_gate_3x.size(0), -1)
        ag_gate_3x = self.ag_gate(ag_gate_3x).view(ag_gate_3x.size(0), -1, 1, 1)
        
        # ag_gate 4xLite
        ag_gate_4x = self.gap(lite4x)
        ag_gate_4x = ag_gate_4x.view(ag_gate_4x.size(0), -1)
        ag_gate_4x = self.ag_gate(ag_gate_4x).view(ag_gate_4x.size(0), -1, 1, 1)
        
        # scaled_stream
        scaled_stream_1x = lite1x * ag_gate_1x
        scaled_stream_2x = lite2x * ag_gate_2x
        scaled_stream_3x = lite3x * ag_gate_3x
        scaled_stream_4x = lite4x * ag_gate_4x
        
        lite_output = scaled_stream_1x + scaled_stream_2x + scaled_stream_3x + scaled_stream_4x
        lite_output = self.conv1x1_up(lite_output)
        
        # residual line
        if self.rescaled_identity:
            identity = self.conv_identity(identity)
            
        x = identity + lite_output
        x = self.relu(self.bn(x))
        
        return x

class OSNet(nn.Module):
    def __init__(self, in_channels, layers, num_classes, feature_extraction=False, fc_dims=512):
        super(OSNet, self).__init__()
        self.feature_extraction = feature_extraction
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=in_channels,
                out_channels=layers[0],
                kernel_size=7,
                stride=2,
                padding=3
            ),
            nn.MaxPool2d(
                kernel_size=3,
                stride=2,
                padding=1
            )
        )
        
        self.conv2 = nn.Sequential(
            BottleNeck(
                in_chan=layers[0],
                out_chan=layers[1],
                rescaled_identity=True
            ),
            BottleNeck(
                in_chan=layers[1],
                out_chan=layers[1]
            )
        )
        
        self.transition_1 = nn.Sequential(
            nn.Conv2d(
                in_channels=layers[1],
                out_channels=layers[1],
                kernel_size=1
            ),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )
        
        self.conv3 = nn.Sequential(
            BottleNeck(
                in_chan=layers[1],
                out_chan=layers[2],
                rescaled_identity=True
            ),
            BottleNeck(
                in_chan=layers[2],
                out_chan=layers[2]
            )
        )
        
        self.transition_2 = nn.Sequential(
            nn.Conv2d(
                in_channels=layers[2],
                out_channels=layers[2],
                kernel_size=1
            ),
            nn.AvgPool2d(
                kernel_size=2,
                stride=2
            )
        )
        
        self.conv4 = nn.Sequential(
            BottleNeck(
                in_chan=layers[2],
                out_chan=layers[3],
                rescaled_identity=True
            ),
            BottleNeck(
                in_chan=layers[3],
                out_chan=layers[3]
            )
        )
        
        self.conv5 = nn.Conv2d(
            in_channels=layers[3],
            out_channels=layers[3],
            kernel_size=1
        )
        
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc = nn.Sequential(
            nn.Linear(layers[3], fc_dims),
            nn.BatchNorm1d(fc_dims),
            nn.ReLU()
        )
        
        self.classifer = nn.Linear(fc_dims, num_classes)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.transition_1(x)
        x = self.conv3(x)
        x = self.transition_2(x)
        x = self.conv4(x)
        x = self.conv5(x)
        x = self.gap(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        if self.feature_extraction:
            return x
        
        x = self.classifer(x)
        return x
    
def OSNet_model(input, num_classes, feature_extraction=False, in_channels=3, layers=[64, 256, 384, 512]):
    model = OSNet(in_channels=in_channels, layers=layers, num_classes=num_classes, feature_extraction=feature_extraction)
    label = model(input)
    return label
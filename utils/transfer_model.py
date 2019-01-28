import torch
import torch.nn as nn
import torchvision

class NTLModel(nn.Module):
    def __init__(self, model, n_classes=3, input_size=(3, 400, 400)):
        super(NTLModel, self).__init__()
        
        self.modelName = 'NTLCNN'
        self.n_classes = n_classes
        self.input_size = input_size
        
        # Convert fc layers to conv layers
        self.features = model.features
        self.classifier = model.classifier
        self.convert_fc_to_conv()
        
        # Freeze conv weights
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        x = self.features(x)  
        x = self.classifier(x)
        return x
    
    def convert_fc_to_conv(self):
        # Create a dummy input tensor and add a dim for batch-size
        x = torch.zeros(self.input_size).unsqueeze_(dim=0)

        # Change last layer output to the num_classes
        self.classifier[-1] = nn.Linear(in_features=self.classifier[-1].in_features,
                                        out_features=self.n_classes)

        # Pass dummy input tensor through features layer to compute output size
        for layer in self.features:
            x = layer(x)

        conv_classifier = []
        kernel_sizes = [(6, 6), (1,1), (1,1)]
        strides = [6, 1 , 1]
        s, k = 0, 0
        for layer in self.classifier:
            if isinstance(layer, nn.Linear):
                # Create a conv equivalent of fc layer
                conv_layer = nn.Conv2d(in_channels=x.size(1),
                                       out_channels=layer.weight.size(0),
                                       kernel_size=kernel_sizes[k],
                                       stride=strides[s],
                                       bias=True)

                # Randomly initialize weights
                torch.nn.init.xavier_uniform_(conv_layer.weight)
                layer = conv_layer
                k += 1
                s += 1
                
                if len(conv_classifier) == 6:
                    avg_pool = nn.AvgPool2d(2)
                    conv_classifier.extend([avg_pool])

            x = layer(x)
            conv_classifier.append(layer)

        # Add average pooling and softmax layer
        softmax = nn.Softmax()
        conv_classifier.extend([softmax])
        self.classifier = nn.Sequential(*conv_classifier)
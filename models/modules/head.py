import torch
from torch import nn
import torch.nn.functional as F

def l2_norm(input, axis=1):
  norm = torch.norm(input, 2, axis, True)
  output = torch.div(input, norm)
  return output


class Slot_Projection(torch.nn.Module):
    def __init__(self, head_type = 'cls'):
        super(Slot_Projection, self).__init__()
        if head_type == 'cls':
            self.non_linear_projection = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.Linear(256, 512),
            )
        elif head_type == 'sim':
            self.non_linear_projection = nn.Sequential(
                nn.Linear(512, 256),
                nn.GELU(),
                nn.LayerNorm(256),
                nn.Linear(256, 512),
            )

    def forward(self, x):
        return self.non_linear_projection(x)

# classifier
class Classifier(nn.Module):

    def __init__(self, width):
        super(Classifier, self).__init__()
        self.classifier_layer = nn.Linear(width, 2)
        self.classifier_layer.weight.data.normal_(0, 0.01)
        self.classifier_layer.bias.data.fill_(0.0)

    def forward(self, input, norm_flag=True):
        if (norm_flag):
              self.classifier_layer.weight.data = l2_norm(
                  self.classifier_layer.weight, axis=-1)
              classifier_out = self.classifier_layer(input)
        else:
              classifier_out = self.classifier_layer(input)
        return classifier_out

class Projection(torch.nn.Module):
    def __init__(self):
        super(Projection, self).__init__()

        self.non_linear_projection = nn.Sequential(
            nn.Linear(768, 512),
            nn.GELU(),
            nn.Linear(512, 512),
        )

    def forward(self, x):
        return self.non_linear_projection(x)


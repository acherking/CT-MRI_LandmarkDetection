import torch
torch.manual_seed(12345)

from torch import nn
import dsntnn

class FCN(nn.Module):
    def __init__(self):
        super().__init__()

        self.layers = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.Conv2d(16, 16, kernel_size=3, padding=1),
        )

    def forward(self, x):
        return self.layers(x)


class CoordRegressionNetwork(nn.Module):
    def __init__(self, n_locations):
        super().__init__()
        self.fcn = FCN()
        self.hm_conv = nn.Conv2d(16, n_locations, kernel_size=1, bias=False)

    def forward(self, images):
        # 1. Run the images through our FCN
        fcn_out = self.fcn(images)
        # 2. Use a 1x1 conv to get one unnormalized heatmap per location
        unnormalized_heatmaps = self.hm_conv(fcn_out)
        # 3. Normalize the heatmaps
        heatmaps = dsntnn.flat_softmax(unnormalized_heatmaps)
        # 4. Calculate the coordinates
        coords = dsntnn.dsnt(heatmaps)

        return coords, heatmaps


from torch import optim
import scipy.misc
import numpy
from PIL import Image

image_size = [40, 40]
raccoon_face = numpy.array(Image.fromarray(scipy.misc.face()[200:400, 600:800, :]).resize(image_size))
eye_x, eye_y = 24, 26

raccoon_face_tensor = torch.from_numpy(raccoon_face).permute(2, 0, 1).float()
input_tensor = raccoon_face_tensor.div(255).unsqueeze(0)
input_var = input_tensor.cuda()

eye_coords_tensor = torch.Tensor([[[eye_x, eye_y]]])
target_tensor = (eye_coords_tensor * 2 + 1) / torch.Tensor(image_size) - 1
target_var = target_tensor.cuda()

print('Target: {:0.4f}, {:0.4f}'.format(*list(target_tensor.squeeze())))

model = CoordRegressionNetwork(n_locations=1).cuda()

optimizer = optim.RMSprop(model.parameters(), lr=2.5e-4)

for i in range(400):
    # Forward pass
    coords, heatmaps = model(input_var)

    # Per-location euclidean losses
    euc_losses = dsntnn.euclidean_losses(coords, target_var)
    # Per-location regularization losses
    reg_losses = dsntnn.js_reg_losses(heatmaps, target_var, sigma_t=1.0)
    # Combine losses into an overall loss
    loss = dsntnn.average_loss(euc_losses + reg_losses)

    # Calculate gradients
    optimizer.zero_grad()
    reg_losses.backward()

    # Update model parameters with RMSprop
    optimizer.step()

# Predictions after training
print('Predicted coords: {:0.4f}, {:0.4f}'.format(*list(coords[0, 0])))
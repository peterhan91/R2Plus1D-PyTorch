import torch
import numpy as np

class IntegratedGradients():
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.model.eval()
        self.hook_layers()

    def hook_layers(self):
        def hook_function(module, grad_in, grad_out):
            self.gradients = grad_in[0]
        # Register hook to the first layer
        first_layer = list(self.model.children())[0]
        first_layer.register_backward_hook(hook_function)

    def generate_images_on_linear_path(self, input_image, steps):
        # Generate uniform numbers between 0 and 1 with total number of steps
        # input_image shape: [1 (batchsize), 3, 32, 224, 224]
        step_list = np.arange(steps+1)/steps
        return [input_image*step for step in step_list]

    def generate_gradients(self, input_image, target_class):
        model_output = self.model(input_image)
        self.model.zero_grad()
        one_hot_output = torch.FloatTensor(1, model_output.size()[-1]).zero_().cuda()
        for n in range(len(target_class)):
            one_hot_output[0][target_class[n]] = 1
        model_output.backward(gradient=one_hot_output)
        gradients_as_arr = self.gradients.cpu().data.numpy()[0]
        return gradients_as_arr

    def generate_integrated_gradients(self, input_image, target_class, steps):
        xbar_list = self.generate_images_on_linear_path(input_image, steps)
        integrated_grads = np.zeros(input_image.size())
        for xbar_image in xbar_list:
            single_integrated_grad = self.generate_gradients(xbar_image, target_class)
            integrated_grads = integrated_grads + single_integrated_grad/steps
        return integrated_grads[0]        # [0] to get rid of the first channel (1,3,32,224,224)
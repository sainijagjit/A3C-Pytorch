import math

import torch
import torch.optim as optim


class SharedAdam(optim.Adam):
    """Implements Adam algorithm with shared states.
    """

    def __init__(self,
                 params,
                 lr=1e-3,
                 betas=(0.9, 0.999),
                 eps=1e-8,
                 weight_decay=0):
        #Initializing adam witht the following parameters
        #params : weights to oprimize 
        #lr : learning rate (default: 1e-3)
        #betas: coefficients used for computing running averages of gradient and its square (default: (0.9, 0.999))
        #eps : term added to the denominator to improve numerical stability (default: 1e-8)
        #weight_decay :  (L2 penalty) (default: 0) lambda * Sum[i=0:n [theta^2]]
        super(SharedAdam, self).__init__(params, lr, betas, eps, weight_decay)

        # Param group of Optimizer
        # This can be useful when fine tuning a pre-trained network as frozen layers can be made trainable and added to the Optimizer as training progresses.
        #param_group (dict) – Specifies what Tensors should be optimized along with group
        # a dict containing all parameter groups
        for group in self.param_groups:
            for p in group['params']:
                #state : a dict holding current optimization state.
                state = self.state[p]
                #initialize step variables to zero
                state['step'] = torch.zeros(1)
                #initialize exp_avg variables to zero
                #Vdw = B1*(Vdw)+(1-B1)dw
                state['exp_avg'] = p.data.new().resize_as_(p.data).zero_()
                #initialize exp_avg_sq variables to zero
                #Sdw = B2*(Sdw)+(1-B2)dw^2
                state['exp_avg_sq'] = p.data.new().resize_as_(p.data).zero_()


    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                #Moves the underlying storage to shared memory.
                #This is a no-operation if the underlying storage is already in shared memory
                #and for CUDA tensors. Tensors in shared memory cannot be resized.
                state['step'].share_memory_()
                #Vdw in shared memory
                state['exp_avg'].share_memory_()
                #Sdw in shared memory
                state['exp_avg_sq'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        
        def closure():
            optimizer.zero_grad()
            output = model(input)
            loss = loss_fn(output, target)
            loss.backward()
            return loss
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                state = self.state[p]

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1

                if group['weight_decay'] != 0:
                    #add(other, *, alpha=1) → Tensor
                    #Add a scalar or tensor to self tensor.
                    #If both alpha and other are specified, each element of other is scaled by
                    #alpha before being used.
                    #grad + wd*w
                    grad = grad.add(group['weight_decay'], p.data)

                #Decay the first and second moment running average coefficient
                #moving_avg = beta1 * moving_avg + (1-beta1) * grad
                #beta1*exp_avg +(1-beta1)*grad
                exp_avg.mul_(beta1).add_(1 - beta1, grad)
                #torch.addcmul(input, tensor1, tensor2, *, value=1, out=None)
                #out<i> = input<i> + value * tensor1<i> * tensor2<i>
                #beta2*exp_avg_sq + (1-beta2)*grad^2
                exp_avg_sq.mul_(beta2).addcmul_(1 - beta2, grad, grad)

                #(exp_avg_sq)^(1/2)+ E => to prevent divide by zero error
                denom = exp_avg_sq.sqrt().add_(group['eps'])

                #(1 - Beta1^t)
                bias_correction1 = 1 - beta1 ** state['step'].item()
                #(1 - Beta2^t)
                bias_correction2 = 1 - beta2 ** state['step'].item()
    
                # alpha * bias_correction2^(1/2)/
                step_size = group['lr'] * math.sqrt(bias_correction2) / bias_correction1

                #torch.addcdiv(input, tensor1, tensor2, *, value=1, out=None) → Tensor
                #out = input + value * (tensor1/tensor2)
                #w + step_size * Vdw<corr>/(Sdw<corr>+E)^(1/2)
                p.data.addcdiv_(-step_size, exp_avg, denom)

        return loss

classdef DiagonalLayer < Layer
    properties
        Sigma
        dL_dSigma
        lambda
        alpha_scale
    end
    
    methods
        function dl = DiagonalLayer(n, std_dev, lambda, alpha_scale)
            dl.Sigma = 1 + std_dev * randn(n, 1);
            dl.lambda = lambda;
            dl.alpha_scale = alpha_scale;
            dl.state = States.FORWARD;
        end
       
        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)
            
            self.X = X;
            self.Y = repmat(self.Sigma, 1, size(X, 2)) .* X;
           
            self.state = States.BACKWARD;
        end
       
        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)
            
            self.dL_dX = repmat(self.Sigma, 1, size(self.X, 2)) .* dL_dY;
            self.dL_dSigma = self.X .* dL_dY;
            self.dL_dSigma = sum(self.dL_dSigma, 2) / size(self.X, 2);
            
            self.X = [];
            self.Y = [];
            self.state = States.UPDATE;
        end
        
        function update(self, alpha)
            States.check_state(self.state, States.UPDATE)
            
            self.Sigma = self.Sigma - alpha * self.alpha_scale * (self.dL_dSigma + self.lambda * (self.Sigma - ones(size(self.Sigma))));
            
            self.dL_dX = [];
            self.dL_dSigma = [];
            self.state = States.FORWARD;
        end
    end
end
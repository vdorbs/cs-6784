classdef DiagonalLayer < Layer
    properties
        Sigma
        dL_dSigma
    end
    
    methods
        function dl = DiagonalLayer(n, std_dev)
            dl.Sigma = 1 + std_dev * randn(n, 1);
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
            
            self.Sigma = self.Sigma - alpha * self.dL_dSigma;
            
            self.dL_dX = [];
            self.dL_dSigma = [];
            self.state = States.FORWARD;
        end
    end
end
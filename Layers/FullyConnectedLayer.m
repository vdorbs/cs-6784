classdef FullyConnectedLayer < Layer
    properties
        W
        dL_dW
    end
    
    methods
        function fcl = FullyConnectedLayer(m, n, sigma)
            fcl.W = sigma * randn(m, n + 1);
            fcl.state = States.FORWARD;
        end
        
        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)
            
            X = [X; ones(1, size(X, 2))];
            self.X = X;
            self.Y = self.W * self.X;
            
            self.state = States.BACKWARD;
        end
        
        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)
            
            self.dL_dX = self.W' * dL_dY;
            self.dL_dX = self.dL_dX(1:end - 1, :);
            self.dL_dW = dL_dY * self.X' / size(self.X, 2);
            
            self.X = [];
            self.Y = [];
            self.state = States.UPDATE;
        end
        
        function update(self, alpha)
            States.check_state(self.state, States.UPDATE)
            
            self.W = self.W - alpha * self.dL_dW;
            
            self.dL_dW = [];
            self.dL_dX = [];
            self.state = States.FORWARD;
        end
    end
end
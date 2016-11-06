classdef BiasLayer < Layer
    properties
        b
        dL_db
        alpha_scale
    end
    
    methods
        function bl = BiasLayer(n, sigma, alpha_scale)
            bl.b = sigma * randn(n, 1);
            bl.alpha_scale = alpha_scale;
            bl.state = States.FORWARD;
        end
       
        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)
           
            self.X = X;
            self.Y = X + repmat(self.b, 1, size(X, 2));
            
            self.state = States.BACKWARD;
        end
       
        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)
            
            self.dL_dX = dL_dY;
            self.dL_db = sum(dL_dY, 2) / size(self.X, 2);
            
            self.X = [];
            self.Y = [];
            self.state = States.UPDATE;
        end
        
        function update(self, alpha)
            States.check_state(self.state, States.UPDATE)
            
            self.b = self.b - alpha * self.alpha_scale * self.dL_db;
            
            self.dL_db = [];
            self.dL_dX = [];
            self.state = States.FORWARD;
        end
    end
end
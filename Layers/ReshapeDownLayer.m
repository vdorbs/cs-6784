classdef ReshapeDownLayer < Layer
    properties
        reshaped_size
    end
    
    methods
        function rdl = ReshapeDownLayer(reshaped_size)
            rdl.reshaped_size = reshaped_size;
            rdl.state = States.FORWARD;
        end
        
        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)
            
            self.X = X;
            self.Y = X(1:self.reshaped_size - 1, :); 
            self.Y = [self.Y; sqrt(sum(X(self.reshaped_size:end, :).^2))];
            
            self.state = States.BACKWARD;
        end
        
        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)
            
            self.dL_dX = dL_dY(1:self.reshaped_size - 1, :);
            self.dL_dX = [self.dL_dX; self.X(self.reshaped_size:end, :)];
            self.dL_dX(self.reshaped_size:end, :) = self.dL_dX(self.reshaped_size:end, :) ./ repmat(sqrt(sum(self.X(self.reshaped_size:end, :).^2)), size(self.X, 1)-self.reshaped_size+1, 1);
            self.dL_dX(self.reshaped_size:end, :) = self.dL_dX(self.reshaped_size:end, :) .* repmat(dL_dY(self.reshaped_size, :), size(self.X, 1)-self.reshaped_size+1, 1);
            
            self.X = [];
            self.Y = [];
            self.state = States.UPDATE;
        end
        
        function update(self, ~)
            States.check_state(self.state, States.UPDATE)
            
            self.dL_dX = [];
            self.state = States.FORWARD;
        end
    end
end
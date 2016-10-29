classdef SoftmaxLoss < Loss
    methods
        function sl = SoftmaxLoss
            sl.state = States.LOSS;
        end
        
        function compute_loss(self, X, Y)
            States.check_state(self.state, States.LOSS)
            
            self.X = X;
            Y = Y + size(X, 1) * (0:size(X, 2) - 1);
            self.Y = Y;
            self.L = sum(-X(Y) + log(sum(exp(X))));
            
            self.state = States.DERIVATIVE;
        end
        
        function differentiate_loss(self)
            States.check_state(self.state, States.DERIVATIVE)
            
            self.dL_dX = zeros(size(self.X));
            self.dL_dX(self.Y) = -1;
            self.dL_dX = self.dL_dX + exp(self.X) ./ repmat(sum(exp(self.X)), size(self.X, 1), 1);
            
            self.X = [];
            self.Y = [];
            self.L = [];
            self.state = States.LOSS;
        end
    end
end
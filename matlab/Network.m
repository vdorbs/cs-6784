classdef Network < handle
    properties
        layers
        loss
        state
        scores
    end
    
    methods
        function network = Network(layers, loss)
            network.layers = layers;
            network.loss = loss;
            network.state = States.FORWARD;
        end
        
        function forward_pass(self, X)
            States.check_state(self.state, States.FORWARD)
            
            for i = 1:length(self.layers)
                self.layers(i).forward_pass(X)
                X = self.layers(i).Y;
            end
            
            self.state = States.LOSS;
        end
        
        function compute_loss(self, X, Y)
            States.check_state(self.state, States.LOSS)
            
            self.loss.compute_loss(X, Y)
            
            self.state = States.DERIVATIVE;
        end
        
        function differentiate_loss(self)
            States.check_state(self.state, States.DERIVATIVE)
            
            self.loss.differentiate_loss
            
            self.state = States.BACKWARD;
        end
            
        function backward_pass(self, dL_dY)
            States.check_state(self.state, States.BACKWARD)
            
            for i = length(self.layers):-1:1
                self.layers(i).backward_pass(dL_dY)
                dL_dY = self.layers(i).dL_dX;
            end
            
            self.state = States.UPDATE;
        end
        
        function update(self, alpha)
            States.check_state(self.state, States.UPDATE)
            
            for i = 1:length(self.layers)
                self.layers(i).update(alpha)
            end
            
            self.state = States.FORWARD;
        end
        
        function losses = train(self, data, alphas, batch_size)
            epochs = length(alphas);
            X = data(:, 1:end - 1)';
            Y = data(:, end)';
            
            losses = zeros(1, epochs * floor(size(X, 2) / batch_size));
            iteration = 1;
            
            epoch = 0;
            for alpha = alphas
                epoch = epoch + 1
                if batch_size == size(X, 2)
                    indices = 1:size(X, 2);
                else
                    indices = randperm(size(X, 2));
                    indices = indices(1:(floor(length(indices) / batch_size) * batch_size));
                    indices = reshape(indices, [length(indices) / batch_size, batch_size]);
                end
                
                for j = 1:size(indices, 1)
                    X_batch = X(:, indices(j, :));
                    Y_batch = Y(indices(j, :));
                    
                    self.forward_pass(X_batch)
                    self.scores = self.layers(end).Y;
                    self.compute_loss(self.scores, Y_batch)
                    losses(iteration) = self.loss.L;
                    iteration = iteration + 1;
                    self.differentiate_loss
                    self.backward_pass(self.loss.dL_dX)
                    self.update(alpha)
                end
            end
        end
        
        function scores = score(self, data)
            self.train(data, 0, size(data, 1));
            scores = self.scores;
        end
        
        function classifications = classify(self, data)
            scores = self.score(data);
            [~, classifications] = max(scores);
        end
    end
end
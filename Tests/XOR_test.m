function XOR_test
    close all
    N = 250;
    X = rand(N, 2);
    Y = 1 + arrayfun(@(x_1, x_2) xor(round(x_1), round(x_2)), X(:, 1), X(:, 2));
    data = [X Y];
    alpha = 1e-3;
    batch_size = N/50;
    epochs = 200;
    sigma = 1;
    hidden_layer_size = 5;
    layers = [ ...
        FullyConnectedLayer(hidden_layer_size, 2, sigma);
        AbsLayer; 
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer; 
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer; 
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer; 
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer; 
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer; 
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer; 
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 2*pi);
        AbsLayer;
        FullyConnectedLayer(2, hidden_layer_size, sigma)];
    loss = SoftmaxLoss;
    network = Network(layers, loss);
    losses = network.train(data, alpha, batch_size, epochs);
    figure
%     semilogy(losses)
    plot(losses);
    title('Normalized loss by iteration')
    classifications = network.classify(data);
    show_classifications(X, classifications)
    
end

function show_classifications(X, classifications)
    falses = X(classifications == 1, :);
    trues = X(classifications == 2, :);
    figure
    hold on
    scatter(falses(:, 1), falses(:, 2), 'r')
    scatter(trues(:, 1), trues(:, 2), 'g')
    title('XOR Classifications')
    legend('False', 'True')
    hold off
end
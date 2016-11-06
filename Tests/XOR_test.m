function XOR_test
    close all
    N = 1000;
    X = rand(N, 2);
    Y = 1 + arrayfun(@(x_1, x_2) xor(round(x_1), round(x_2)), X(:, 1), X(:, 2));
    data = [X Y];
    epochs = 10;
    alphas = logspace(-1, -2, epochs);
    batch_size = N / 10;
    sigma = 1;
    lambda = 1e-2;
    hidden_layer_size = 25;
    layers = [ ...
        ReshapeUpLayer(hidden_layer_size);
        DizzyLayer(hidden_layer_size, 1);
        DiagonalLayer(hidden_layer_size, sigma, lambda, 1);
        DizzyLayer(hidden_layer_size, 1);
        BiasLayer(hidden_layer_size, sigma, 1);
%         FullyConnectedLayer(hidden_layer_size, 2, sigma, 1);
        ReLULayer;
        FullyConnectedLayer(2, hidden_layer_size, sigma, 1)];
    
    loss = SoftmaxLoss;
    network = Network(layers, loss);
    losses = network.train(data, alphas, batch_size);
    figure
%     semilogy(losses)
    plot(losses);
    title('Normalized unregularized loss by iteration')
    classifications = network.classify(data);
    show_classifications(X, classifications)
    scores = network.score(data);
    figure
    scatter3(X(:, 1), X(:, 2), scores(1, :)')
    title('Scores - false')
    xlabel('x')
    ylabel('y')
    zlabel('z')
    figure
    scatter3(X(:, 1), X(:, 2), scores(2, :)')
    title('Scores - true')
    xlabel('x')
    ylabel('y')
    zlabel('z')
    
end

function show_classifications(X, classifications)
    falses = X(classifications == 1, :);
    trues = X(classifications == 2, :);
    figure
    hold on
    scatter(falses(:, 1), falses(:, 2), 'r')
    scatter(trues(:, 1), trues(:, 2), 'g')
    grid on
    title('XOR Classifications')
    legend('False', 'True')
    hold off
end
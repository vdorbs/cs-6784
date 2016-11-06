function XOR_test
    close all
    N = 1000;
    X = rand(N, 2);
    Y = 1 + arrayfun(@(x_1, x_2) xor(round(x_1), round(x_2)), X(:, 1), X(:, 2));
    data = [X Y];
    epochs = 250;
    alphas = logspace(-1, -3, epochs);
    batch_size = N / 10;
    sigma = 1;
    lambda = 1e-2;
    hidden_layer_size = 5;
    layers = [ ...
        ReshapeUpLayer(hidden_layer_size);
        DizzyLayer(hidden_layer_size, 1);
        DiagonalLayer(hidden_layer_size, sigma, lambda, 1);
        DizzyLayer(hidden_layer_size, 1);
        BiasLayer(hidden_layer_size, sigma, 1);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 1);
        DiagonalLayer(hidden_layer_size, sigma, lambda, 1);
        DizzyLayer(hidden_layer_size, 1);
        BiasLayer(hidden_layer_size, sigma, 1);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 1);
        DiagonalLayer(hidden_layer_size, sigma, lambda, 1);
        DizzyLayer(hidden_layer_size, 1);
        BiasLayer(hidden_layer_size, sigma, 1);
        AbsLayer;
        DizzyLayer(hidden_layer_size, 1);
        DiagonalLayer(hidden_layer_size, sigma, lambda, 1);
        DizzyLayer(hidden_layer_size, 1);
        BiasLayer(hidden_layer_size, sigma, 1);
        AbsLayer;
        FullyConnectedLayer(2, hidden_layer_size, sigma, 0)];
    loss = SoftmaxLoss;
    network = Network(layers, loss);
    losses = network.train(data, alphas, batch_size);
    subplot(2, 2, 1)
    plot(losses);
    title('Normalized unregularized loss by iteration')
    classifications = network.classify(data);
    falses = X(classifications == 1, :);
    trues = X(classifications == 2, :);
    subplot(2, 2, 2)
    hold on
    scatter(falses(:, 1), falses(:, 2), 'r')
    scatter(trues(:, 1), trues(:, 2), 'g')
    grid on
    title('XOR Classifications')
    legend('False', 'True')
    hold off
    scores = network.score(data);
    subplot(2, 2, 3)
    scatter3(X(:, 1), X(:, 2), scores(1, :)')
    title('Scores - false')
    xlabel('x')
    ylabel('y')
    zlabel('z')
    subplot(2, 2, 4)
    scatter3(X(:, 1), X(:, 2), scores(2, :)')
    title('Scores - true')
    xlabel('x')
    ylabel('y')
    zlabel('z')
end
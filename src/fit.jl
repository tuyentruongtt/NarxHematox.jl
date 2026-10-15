using Optimisers

"""
    fit_narx!(model, x, y, loss; lr, max_epochs, early_stopping,
              validation = nothing, reg = m -> 0f0, prune = nothing, frozen = (),
              decay_steps = 1800, decay_rate = 0.8, verbose = false)

Train `model` on inputs `x` and targets `y` with Adam and exponential learning-rate decay,
one full-batch step per epoch, as in the original Keras setup.

- `loss(y_true, y_pred)`: loss function, e.g. `mse_missing`.
- `early_stopping`: an `EarlyStoppingAveraged`.
- `validation`: optional `(x_val, y_val)`; if given, the validation loss is monitored.
- `reg(model)`: weight penalty added to the loss, e.g. `m -> regularization(m, 0.01)`.
- `prune`: optional `Prune`, applied after each step.
- `frozen`: names of fields not to train, e.g. `(:cell,)`.

Returns the vector of monitored losses.
"""
function fit_narx!(model, x, y, loss; lr, max_epochs, early_stopping,
                   validation = nothing, reg = m -> 0f0, prune = nothing, frozen = (),
                   decay_steps = 1800, decay_rate = 0.8, verbose = false)
    opt = Flux.setup(Optimisers.Adam(lr, (0.9, 0.999), 1e-7), model)
    for name in frozen
        Optimisers.freeze!(getproperty(opt, name))
    end

    history = Float64[]
    on_train_begin!(early_stopping, model)

    for epoch in 1:max_epochs
        Optimisers.adjust!(opt, lr * decay_rate^((epoch - 1) / decay_steps))
        l, grads = Flux.withgradient(m -> loss(y, m(x)) + reg(m), model)
        Flux.update!(opt, model, grads[1])
        isnothing(prune) || prune!(model, prune)
        model.alpha .= model.constraint(model.alpha)

        monitored = isnothing(validation) ? l :
                    loss(validation[2], model(validation[1])) + reg(model)
        push!(history, monitored)

        verbose && epoch % 100 == 0 && println("epoch $epoch: loss $(round(monitored; sigdigits = 4))")
        on_epoch_end!(early_stopping, model, history) && break
    end

    return history
end
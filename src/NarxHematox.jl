module NarxHematox

using Flux

include("friberg.jl")
include("narx.jl")
include("utils.jl")
include("weights.jl")
include("fit.jl")

export solve_friberg, friberg, poplog_fit_friberg_mse, test_friberg
export TFGRUCell, ARX_RNN_ES, ARX_net_lagged
export InputNoise, Prune, LP_regularizer, prune!, regularization
export mse_missing, step_mse_missing, smoothing_penalty, smse_smoothing, combined_loss
export EarlyStoppingAveraged, on_train_begin!, on_epoch_end!
export buildrnn, buildmlp, buildNARX
export index_weights_path, load_weights!
export fit_narx!
export with_input_noise

end
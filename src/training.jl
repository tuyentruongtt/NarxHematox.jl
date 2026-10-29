import Optim

const P0 = [0.316, 195.0, 2.0, 270.0]
const LOWER_B = [0.05, 20.0, 1.0, 25.0]
const UPPER_B = [0.45, 350.0, 5.0, 450.0]

"""
    default_hp(celltype)

Hyperparameters of the original showcase for `"GRU"` or `"FNN"`.
"""
function default_hp(celltype)
    return Dict{Symbol,Any}(
        :xlag => 28, :ylag => 21, :lag => 28,
        :units => celltype == "GRU" ? 64 : 4,
        :activation => "sigmoid", :out_act => "linear", :out_bias => false, :es => true,
        :stddev => 0.0, :stddev_ft => 0.2,
        :k => 2, :c => 1e-2, :s => 1.0, :s_train => 0.1,
        :delta => 0.007, :delta_ft => celltype == "GRU" ? 0.007 : 0.005,
        :lp => 1e-4, :out_l2 => 0.01,
        :patience => 250, :max_epochs => 10200, :av_epochs => 20,
        :min_delta_fri => 1e-4, :min_delta_ptr => 1e-4, :min_delta_ft => 1e-5,
        :lr => 0.01, :lr_ft => 0.001,
    )
end

function read_scenarios(path, n)
    lines = filter(!isempty, collect(Iterators.drop(eachline(path), 1)))
    data = reduce(vcat, [permutedims(parse.(Float64, split(line, ','))) for line in lines])
    return [(Int.(data[:, 2i]), data[:, 2i+1]) for i in 1:n]
end

function gru_features(X::AbstractMatrix, Y::AbstractMatrix, lag)
    B, L = size(X)
    feat = zeros(Float32, 2, L, B)
    feat[1, :, :] = permutedims(X)
    feat[2, :, :] = permutedims(Y)
    feat[2, lag+1:end, :] .= NaN32
    target = reshape(Float32.(permutedims(Y[:, lag+1:end])), 1, L - lag, B)
    return feat, target
end

function fnn_features(feat, xlag, ylag)
    L, B = size(feat, 2), size(feat, 3)
    T = L - xlag + 1
    lagged = zeros(Float32, xlag + ylag, T, B)
    for i in 1:T
        lagged[1:ylag, i, :] = feat[2, i:i+ylag-1, :]
        lagged[ylag+1:end, i, :] = feat[1, i:i+xlag-1, :]
    end
    return lagged
end

"""
    calibrate_friberg(t_arr, y_arr, cycle_arr, therapy; train_cycles = 4)

Fit the Friberg parameters to the observations with `cycle_arr .<= train_cycles`
with Nelder-Mead and a population penalty, as in the original showcase.

- `t_arr`: observation days, starting at day 0.
- `y_arr`: platelet counts per litre.
- `cycle_arr`: therapy cycle of each observation (the `CYCLE` column of the data).
- `therapy`: `(therapy days, doses)`, one therapy per cycle.

Returns the Friberg parameters `[gamma, MTT, slope, c0/10^9]`.
"""
function calibrate_friberg(t_arr, y_arr, cycle_arr, therapy; train_cycles = 4)
    mask = cycle_arr .<= train_cycles
    t_c = t_arr[mask]
    y_arr_c = y_arr[mask]
    n_c = Int(maximum(cycle_arr[mask]))
    therapies_c = (therapy[1][1:n_c], therapy[2][1:n_c])
    fit_loss(p) = poplog_fit_friberg_mse(clamp.(p, LOWER_B, UPPER_B), P0, y_arr_c, t_c, therapies_c)
    res = Optim.optimize(fit_loss, P0, Optim.NelderMead(), Optim.Options(g_tol = 1e-2))
    return clamp.(Optim.minimizer(res), LOWER_B, UPPER_B)
end

"""
    train_patient(t_arr, y_arr, cycle_arr, therapy; celltype = "GRU", train_cycles = 4,
                  hp = default_hp(celltype), pc = calibrate_friberg(...), verbose = true)

Calibrate the Friberg model and train an ARX-GRU or ARX-FNN for one patient,
as in the original Python showcase.

- `t_arr`: observation days, starting at day 0.
- `y_arr`: platelet counts per litre.
- `cycle_arr`: therapy cycle of each observation (the `CYCLE` column of the data).
  Observations with `cycle_arr .<= train_cycles` are used for training, the rest for testing.
- `therapy`: `(therapy days, doses)`, one therapy per cycle.
- `pc`: Friberg parameters; calibrated with `calibrate_friberg` if not given.

Returns a named tuple with the daily time grid `t`, the predictions `y_fri` and `y_narx`,
the Friberg parameters `pc`, the test errors `smse_fri` and `smse_narx`,
the first test day `split_day`, and the trained `model`.
"""
function train_patient(t_arr, y_arr, cycle_arr, therapy; celltype = "GRU", train_cycles = 4,
                       hp = default_hp(celltype),
                       pc = calibrate_friberg(t_arr, y_arr, cycle_arr, therapy; train_cycles = train_cycles),
                       verbose = true)
    hpx = merge(hp, Dict(:stddev => hp[:stddev_ft], :delta => hp[:delta_ft]))
    t = collect(0:t_arr[end])

    mask = cycle_arr .<= train_cycles
    t_c = t_arr[mask]
    y_arr_c = y_arr[mask]
    y_fri = friberg(pc, t, therapy)

    scale_lo, scale_hi = celltype == "GRU" ? (-0.5, 0.5) : (-1.0, 1.5)
    log_lo, log_hi = log10(25e9), log10(300e9)
    to_scaled(y) = scale_lo .+ (log10.(y) .- log_lo) .* (scale_hi - scale_lo) ./ (log_hi - log_lo)
    from_scaled(s) = 10 .^ (log_lo .+ (s .- scale_lo) .* (log_hi - log_lo) ./ (scale_hi - scale_lo))
    stst = to_scaled(y_arr_c[1])

    scenarios = vcat([therapy], read_scenarios(joinpath(@__DIR__, "..", "data", "therapies_all.csv"), 7))
    t_sc = collect(0:299)
    X_sc = zeros(8, 300)
    Y_sc = zeros(8, 300)
    for (i, (days, doses)) in enumerate(scenarios)
        X_sc[i, days .+ 1] .= doses
        Y_sc[i, :] = to_scaled(friberg(pc, t_sc, (days, doses)))
    end

    lag = hp[:lag]
    X = zeros(length(t))
    X[therapy[1] .+ 1] .= therapy[2]
    y_pad = fill(NaN, length(t))
    y_pad[t_arr .+ 1] = to_scaled(y_arr)
    X_add = vcat(zeros(lag), X)
    y_add = vcat(fill(stst, lag), y_pad)
    split_day = t_c[end] + 1

    X_fr = hcat(zeros(8, lag), X_sc)
    Y_fr = hcat(fill(stst, 8, lag), Y_sc)

    row(v) = reshape(v, 1, :)
    x_ind, y_ind = gru_features(row(X_add[1:split_day+lag]), row(y_add[1:split_day+lag]), lag)
    x_test, _ = gru_features(row(X_add), row(y_add), lag)
    x_fr, y_fr = gru_features(X_fr[1:6, :], Y_fr[1:6, :], lag)
    x_val, y_val = gru_features(X_fr[7:8, :], Y_fr[7:8, :], lag)

    if celltype == "FNN"
        x_ind, x_test, x_fr, x_val = (fnn_features(x, hp[:xlag], hp[:ylag]) for x in (x_ind, x_test, x_fr, x_val))
    end

    build(h) = celltype == "GRU" ? buildrnn(h) : buildNARX(h)
    reg(h) = celltype == "GRU" ? (m -> regularization(m, h[:out_l2])) :
                                 (m -> regularization(m, LP_regularizer(lp = h[:lp], p = 0.5)))
    early(min_delta) = EarlyStoppingAveraged(hp[:av_epochs], hp[:patience]; min_delta = min_delta)
    smse_loss(yt, yp) = smse_smoothing(yt, yp; k = hp[:k], c = hp[:c], s = hp[:s_train])

    verbose && println("Phase 1: transfer learning on Friberg scenarios")
    model = load_weights!(build(hp), index_weights_path(celltype))
    fit_narx!(model, x_fr, y_fr, mse_missing; lr = hp[:lr], max_epochs = hp[:max_epochs],
              early_stopping = early(hp[:min_delta_fri]), validation = (x_val, y_val),
              reg = reg(hp), prune = Prune(hp[:delta]), verbose = verbose)

    verbose && println("Phase 2: fine-tuning the output layer on the patient")
    model_ft = with_input_noise(model, hpx[:stddev])
    fit_narx!(model_ft, x_ind, y_ind, smse_loss; lr = hp[:lr_ft], max_epochs = hp[:max_epochs],
              early_stopping = early(hp[:min_delta_ptr]), reg = reg(hpx), prune = Prune(hpx[:delta]),
              frozen = celltype == "GRU" ? (:cell,) : (), verbose = verbose)

    verbose && println("Phase 3: fine-tuning the full model on the patient")
    fit_narx!(model_ft, x_ind, y_ind, smse_loss; lr = hp[:lr_ft], max_epochs = hp[:max_epochs],
              early_stopping = early(hp[:min_delta_ft]), reg = reg(hpx), prune = Prune(hpx[:delta]),
              verbose = verbose)

    y_narx = from_scaled(vec(model_ft(x_test)))

    ylog_obs = fill(NaN, length(t))
    ylog_obs[t_arr .+ 1] = log10.(y_arr)
    test_idx = (split_day + 1):length(t)
    smse_narx = step_mse_missing(ylog_obs[test_idx], log10.(y_narx)[test_idx], hp[:s])
    smse_fri = step_mse_missing(ylog_obs[test_idx], log10.(y_fri)[test_idx], hp[:s])

    return (; t, y_fri, y_narx, pc, smse_fri, smse_narx, split_day, model = model_ft)
end
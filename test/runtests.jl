using NarxHematox
using Test
using Flux

@testset "Friberg" begin
    pat_p = [0.3222668, 217.17298364, 2.10676328, 276.05649911]
    therapy = (collect(0:5) .* 21, ones(6))
    y = friberg(pat_p, collect(0:124), therapy)
    @test length(y) == 125
    @test y[1] ≈ 2.76056499110e11 rtol = 1e-4
    @test y[31] ≈ 2.142880458786572e11 rtol = 1e-3
    @test y[61] ≈ 1.789882061149882e11 rtol = 1e-3
    @test y[125] ≈ 1.909523362974695e11 rtol = 1e-3
end

@testset "NARX models" begin
    m1 = ARX_RNN_ES(TFGRUCell(2, 8), 5)
    x1 = rand(Float32, 2, 20, 3)
    @test size(m1(x1)) == (1, 15, 3)
    @test Flux.gradient(m -> sum(m(x1)), m1)[1] !== nothing

    m2 = ARX_net_lagged(Chain(Dense(5 => 4, σ), Dense(4 => 1)), 2, 2)
    x2 = rand(Float32, 5, 20, 3)
    @test size(m2(x2)) == (1, 19, 3)
    @test Flux.gradient(m -> sum(m(x2)), m2)[1] !== nothing
end

@testset "Utils" begin
    hp_rnn = Dict(:units => 8, :lag => 5, :stddev => 0.2, :activation => "sigmoid",
                  :out_act => "linear", :out_bias => false)
    m1 = buildrnn(hp_rnn)
    x1 = rand(Float32, 2, 20, 3)
    @test size(m1(x1)) == (1, 15, 3)
    @test m1(x1) == m1(x1)
    @test Flux.gradient(m -> sum(m(x1)) + regularization(m, 0.01), m1)[1] !== nothing

    hp_fnn = Dict(:xlag => 3, :ylag => 2, :units => 4, :stddev => 0.0, :activation => "sigmoid",
                  :out_act => "linear", :out_bias => false, :es => true)
    m2 = buildNARX(hp_fnn)
    x2 = rand(Float32, 5, 20, 3)
    @test size(m2(x2)) == (1, 19, 3)
    @test Flux.gradient(m -> sum(m(x2)) + regularization(m, LP_regularizer(lp = 1e-4, p = 0.5)), m2)[1] !== nothing

    @test mse_missing([1.0, NaN, 3.0], [1.0, 5.0, 4.0]) ≈ 0.5
    @test step_mse_missing([1.0, NaN, 3.0], [1.0, 5.0, 4.0], 1.0) ≈ 3.65 rtol = 1e-6
    @test smoothing_penalty(ones(10); k = 2, c = 1.0) ≈ 0.24

    prune!(m1, Prune(10f0))
    @test all(m1.dense.weight .== 0)

    es = EarlyStoppingAveraged(2, 3)
    on_train_begin!(es, m2)
    history = [5.0, 4.0, 3.0, 3.0, 3.0, 3.0, 3.0, 3.0]
    stop_at = findfirst(n -> on_epoch_end!(es, m2, history[1:n]), 1:length(history))
    @test stop_at == 7
end

@testset "Pretrained weights" begin
    hp_gru = Dict(:units => 64, :lag => 28, :stddev => 0.0, :activation => "sigmoid",
                  :out_act => "linear", :out_bias => false)
    m1 = load_weights!(buildrnn(hp_gru), index_weights_path("GRU"))
    x1 = zeros(Float32, 2, 40, 1)
    for t in 0:39
        x1[1, t+1, 1] = t % 21 == 0 ? 1 : 0
        x1[2, t+1, 1] = Float32(0.5 * cos(0.2 * t))
    end
    ref1 = [0.549197078, 0.538959026, 0.522476852, 0.502165973, 0.479454517, 0.455263197,
            0.430213451, 0.404702157, 0.378944099, 0.353040397, 0.327103496, 0.301459193]
    @test vec(m1(x1)) ≈ ref1 rtol = 1e-4

    hp_fnn = Dict(:xlag => 28, :ylag => 21, :units => 4, :stddev => 0.0, :activation => "sigmoid",
                  :out_act => "linear", :out_bias => false, :es => true)
    m2 = load_weights!(buildNARX(hp_fnn), index_weights_path("FNN"))
    x2 = Float32[0.1 * sin(f + 0.3 * t) for f in 0:48, t in 0:9]
    x2 = reshape(x2, 49, 10, 1)
    ref2 = [1.51173747, 1.61541748, 1.79179192, 1.98638582, 2.14246988,
            2.26011562, 2.35028887, 2.42154694, 2.47878313]
    @test vec(m2(x2)) ≈ ref2 rtol = 1e-4
end

@testset "Training" begin
    hp = Dict(:units => 4, :lag => 3, :stddev => 0.0, :activation => "sigmoid",
              :out_act => "linear", :out_bias => false)
    m = buildrnn(hp)
    x = rand(Float32, 2, 10, 2)
    y = rand(Float32, 1, 7, 2)
    es = EarlyStoppingAveraged(5, 50)
    h = fit_narx!(m, x, y, mse_missing; lr = 0.01, max_epochs = 30, early_stopping = es,
                  reg = m -> regularization(m, 0.01), prune = Prune(0.001f0), frozen = (:cell,))
    @test length(h) == 30
    @test h[end] < h[1]
    @test all(0 .<= m.alpha .<= 1)
end

@testset "Calibration" begin
    pat_p = [0.3222668, 217.17298364, 2.10676328, 276.05649911]
    therapy = (collect(0:5) .* 14, ones(6))
    t_arr = collect(0:2:82)
    cycle_arr = div.(t_arr, 14) .+ 1
    y_arr = friberg(pat_p, t_arr, therapy)
    pc = calibrate_friberg(t_arr, y_arr, cycle_arr, therapy; train_cycles = 4)
    @test length(pc) == 4
    @test all(isfinite, pc)
    @test all(NarxHematox.LOWER_B .<= pc .<= NarxHematox.UPPER_B)
        y_nan = copy(y_arr)
    y_nan[5] = NaN
    keep = [1:4; 6:length(t_arr)]
    therapies_c = (therapy[1][1:4], therapy[2][1:4])
    l_nan = poplog_fit_friberg_mse(pat_p, pat_p, y_nan, t_arr, therapies_c)
    l_drop = poplog_fit_friberg_mse(pat_p, pat_p, y_arr[keep], t_arr[keep], therapies_c)
    @test isfinite(l_nan)
    @test l_nan ≈ l_drop
end
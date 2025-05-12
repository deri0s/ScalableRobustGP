# Zero, 3th expert config
# Standard Moving Average
sgp = DPSGP(X_train, y_train, init_K=7,
            gp_model='Sparse',
            prior_mean=ConstantMean(), kernel=covar_module,
            lengthscale=lss,
            N_iter=15,
            noise_var = 0.06,
            floating_point=floating_point,
            normalise_y=True,
            DP_max_iter=390,
            window_size=40,
            threshold_factor=2.5,
            print_conv=True, plot_conv=True, plot_sol=True)
# Adaptive Moving Average
threshold_factor=2.5,

# 1st experts config
lss = [1.83, 0.8, 603, 0.3, 5.87e+04, 3.0, 2.17, 1.2e+03, 4.63, 1, 1.19e+04, 52.2, 663, 17.3]
sgp = DPSGP(X_train, y_train, init_K=7,
            gp_model='Sparse',
            prior_mean=ConstantMean(), kernel=covar_module,
            lengthscale=lss,
            N_iter=15,
            noise_var = 0.06,
            floating_point=floating_point,
            normalise_y=True,
            DP_max_iter=390,
            window_size=150,
            threshold_factor=1,
            print_conv=True, plot_conv=True, plot_sol=True)
# Adaptive Moving Average
threshold_factor=1.5
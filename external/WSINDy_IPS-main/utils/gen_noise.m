function [U_obs,noise,snr,sigma] = gen_noise(U_exact,sigma_NR,noise_dist,noise_alg,rng_seed,toggle_disp)

    n = length(U_exact);
    rng(rng_seed);

    if sigma_NR>0
        U_obs = cell(n,1);
        stdvs = zeros(1,n);
        noise = cell(1,n);
        snr = zeros(1,n);
        for k=1:n
            stdvs(k) = rms(U_exact{k}(:))^2;
        end
        for j=1:n
            [U_obs{j},noise{j},snr(j),sigma] = add_noise(U_exact{j},stdvs(j),sigma_NR,noise_dist,noise_alg);
            if toggle_disp
                disp(['[SNR sigma] = ',num2str([snr(j) sigma])]); 
            end
        end
    else
        U_obs = U_exact;
        noise = [];
        snr = 0;
        sigma = 0;
    end

end

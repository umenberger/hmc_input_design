function posterior_pdf = hack_2param_posterior_lgss(ops)

	theta_true = ops.theta;
    theta_tmp = theta_true;

	ths = ops.theta_range;
    
    u = ops.u;
    y = ops.y;

	n = size(ths,2);

    pst = zeros(n,n);

    for i = 1:n

        tmpA = theta_true.A;
        tmpA(2,1) = ths(1,i);

        for j = 1:n

            tmpA(2,2) = ths(2,j);
            theta_tmp.A = tmpA;

            pst(i,j) = exp(robust_loglik(y,u,theta_tmp));

        end

    end

    posterior_pdf = zeros(2,n);

    for i = 1:n
        posterior_pdf(1,i) = trapz(ths(2,:),pst(i,:));
    end

    for i = 1:n
        posterior_pdf(2,i) = trapz(ths(1,:),pst(:,i));
    end

    % volume
    v = trapz(ths(1,:),posterior_pdf(1,:));

    npst = trapz(ths(1,:),posterior_pdf(1,:));
    posterior_pdf(1,:) = posterior_pdf(1,:)/npst;

    npst = trapz(ths(2,:),posterior_pdf(2,:));
    posterior_pdf(2,:) = posterior_pdf(2,:)/npst;

%     pst_nom = pst/v;

  end
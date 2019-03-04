function [f,df] = nmpc_hmc_flex_nl_multicost(u,ops)


q_samples = ops.q_samples;
p_samples = ops.p_samples;

num_samples = size(q_samples,2);

f = 0;
df = 0;

ops2 = ops;
ops2.q_samples = [];
ops2.p_samples = [];
ops2.u = u;
ops2.return_gradient = 1;

parfor i = 1:num_samples
    
    [f_,df_] = nmpc_hmc_flex_nl_cost(ops2,q_samples(:,i),p_samples(:,i));
    
    f = f + f_;
    df = df + df_;

end

f = f/num_samples;
df = df/num_samples;



end



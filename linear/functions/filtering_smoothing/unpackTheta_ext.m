function [A,B,C,D,E,G,Q,R,m1,P1,P] = unpackTheta_ext(theta)
% Function to quickly unpack system parameters from structure

    A = theta.A;
    B = theta.B;
    C = theta.C;
    D = theta.D;
    E = theta.E;
    G = theta.G;
    Q = theta.Q;
    R = theta.R;
    m1 = theta.m1;
    P1 = theta.P1;
    P = theta.P;

end


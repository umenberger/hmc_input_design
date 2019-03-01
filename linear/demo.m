%%

% code to reproduce the linear example in the paper:
% "Nonlinear input design as optimal control of a Hamiltonian system"
% by Jack Umenberger and Thomas B. Schon

% Note: code is not optimized for performance.

clear variables
close all
clc

%%

% set to 0 to use data from the paper
% set to 1 to generate a new example
regenerate_data = 1; 

if regenerate_data
    optimize_input; % note: this can be quite slow
else
    load('data_from_paper.mat')
end

generate_figures; 



% FUNCTION [y] = fractionalDelay(x, mu)
%
% PURPOSE: Advace signal in time by mu fractional samples.
%
% INPUT: x, vector signal
%
% OUTPUT: y, vector signal taken at fractional sampling times x(n + mu) 
%
% AUTHOR: Neal Patwari
%
function [y] = fractionalDelay(x, mu)

% Cubic Farrow interpolator
h_I  = [ (1/6)*mu^3            - (1/6)*mu,  ...
        -0.5*  mu^3 + 0.5*mu^2 +       mu,  ...
         0.5*  mu^3 -     mu^2 -  0.5* mu    + 1,  ...
        -(1/6)*mu^3 + 0.5*mu^2 - (1/3)*mu];

% Use Matlab's FIR filter and initialize with zeros.
y_long = filter(h_I, 1, [x 0 0]);
y = y_long(3:end);
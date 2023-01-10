% Purpose: Convert M-ary data to binary data
%          each m-ary value input in "data" is converted to 
%          log2(M) binary values.
% Input:   M-ary digit vector
% Output:  Binary digit vector, with length equal to the number
%          of values in data multiplied by log2(M)
%
function [binarydata] = mary2binary(data, M)

len       = length(data);    % number of values in data
log2M     = round(log2(M));  % integer number of bits per data value

temp = (dec2bin(data,log2M) == '1')';  % Convert each value in data to a row of log2M binary values
binarydata = temp(:)';  % Convert the temp matrix to a vector

% Purpose: Convert binary data to M-ary data by making groups of log2(M)
%          bits and converting each bit to one M-ary digit.
% Input:   Binary digit vector, with length as a multiple of log2(M)
% Output:  M-ary digit vector
%
function [marydata] = binary2mary(data, M)

len       = length(data);
log2M     = round(log2(M));  % integer number of bits per group
if (mod(len,log2M)~=0),
    error('Input to binary2mary must be divisible by log2(m).');
end

binvalues = 2.^(log2M-1:-1:0)';
temp      = reshape(data, log2M, length(data)/log2M)';  
marydata  = (temp * binvalues)';

% Purpose: Find the symbols which are closest in symbol space
%    to the received signal space values.
% Input:   Received r hat values (output of matched filter),
%    and possible signal space values.  Assumes signal space estimate 
%    vectors and signal space vectors are in the columns of matrices 
%    r_hat and outputVec
% Output:  m-ary symbol indices in 0...length(outputVec)-1

function [data_out] = findClosestMary(r_hat, outputVec)

symbols         = size(outputVec,2);
data_out        = ones(1,length(r_hat));
% For each r_hat value
for i = 1:size(r_hat,2),
    % Find the index of the symbol space value with the
    % lowest distance to this r_hat.
    distances   = sum((r_hat(:,i)*ones(1,symbols) - outputVec).^2, 1);
    [val, ind]  = min(distances);
    % Output values 0, 1, ..., length(outputVec)-1
    data_out(i) = ind-1;  
end
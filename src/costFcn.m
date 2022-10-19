function [F, G, H] = costFcn(Z, X, y, lambda, t)
% Compute the cost function F(Z)
%
% INPUTS: 
%   Z: Parameter values
% OUTPUTS
%   F: Function value
%   G: Gradient value
%   H: Hessian value
%

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% To improve the excution speed, please program your code with matrix
% format. It is 30 times faster than the code using the for-loop.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
W = Z.W;
C = Z.C;
zeta = Z.zeta;
sample_len = size(zeta,1);
w_len = size(W, 1);
term = ((W'*X)'.*y + C*y + zeta - ones(sample_len,1));
denominator = 1./ term;
df_zeta = ones(sample_len,1) - (1/t) * denominator - (1/t)*(1./zeta);
df_W = 2*lambda*W - (1/t)*(X.*y'* denominator);
df_C = -(1/t) * y'* denominator;
F = sum(zeta) + lambda*(W'*W) - (1/t)*sum(log(term))-(1/t)*sum(log(zeta));
G = [df_W; df_C; df_zeta];

df_W2 = 2*lambda*eye(w_len) + (X./term')*(X./term')'/t;
df_W_C = 1/t * X * (denominator.^2);
df_C2 = (1/t) * sum(denominator.^2);
df_W_zeta = 1/t * X.*y'./ (term.^2)';
df_C_zeta = 1/t * y'./ (term.^2)';
%df_zeta2 = diag(1/t*(1./(term.^2)'+ 1./(zeta.^2)'));
%df_W_zeta = zeros(w_len, sample_len);
% for j=1:w_len
%     for k=1:sample_len
%         df_W_zeta(j,k) = 1/t * X(j,k) * y(k) / (W'*X(:,k)*y(k)+C*y(k)+zeta(k)-1)^2;
%     end
% end
% df_C_zeta = zeros(1, sample_len);
% for j=1:sample_len
%     df_C_zeta(j) = 1/t*y(j)/(W'*X(:,j)*y(j)+C*y(j)+zeta(j)-1)^2;
% end
df_zeta2 = zeros(sample_len, sample_len);
for j=1:sample_len
    df_zeta2(j,j) = 1/t*(1/(W'*X(:,j)*y(j)+C*y(j)+zeta(j)-1)^2 + 1/zeta(j)^2);
end

H=[df_W2 df_W_C df_W_zeta; df_W_C' df_C2 df_C_zeta; df_W_zeta' df_C_zeta' df_zeta2];

end
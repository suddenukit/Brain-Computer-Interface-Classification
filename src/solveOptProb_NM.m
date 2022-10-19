function [Z, err] = solveOptProb_NM(costFcn,init_Z,tol, X, y, lambda,t)
% Compute the optimal solution using Newton method
%
% INPUTS:
%   costFcn: Function handle of F(Z)
%   init_Z: Initial value of Z
%   tol: Tolerance
%
% OUTPUTS:
%   optSolution: Optimal soultion
%   err: Errorr
%

Z = init_Z;
optSolution = [Z.W; Z.C; Z.zeta];
W_len = size(Z.W,1);
zeta_len = size(Z.zeta,1);
err = 100;

% Set the error 2*tol to make sure the loop runs at least once
while (err/2) > tol
    % Execute the cost function at the current iteration
    % F : function value, G : gradient, H, hessian
    [F, G, H] = costFcn(Z, X, y, lambda, t);
    Hinv = H^-1;
    s = 1;
    err = (G'* Hinv * G);
    %disp(["Err:" num2str(err)]);
    newSol = optSolution - s * Hinv * G;
    while (~all((newSol(1:W_len)'*X)'.*y + newSol(W_len+1)*y + newSol(W_len+2:end) - ones(zeta_len,1)>0) || ~all(newSol(W_len+2:end)>0))
        s = s*0.5;
        newSol = optSolution - s * Hinv * G;
    end
    optSolution = newSol;
    Z.W = optSolution(1:W_len);
    Z.C = optSolution(W_len+1);
    Z.zeta = optSolution(W_len+2:end);
end

end




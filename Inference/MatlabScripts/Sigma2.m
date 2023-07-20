function Sigma2Out = Sigma2(NuIn, Net)
%
%  Sigma2Out = Sigma2(NuIn, Net)
%

if size(NuIn,1)==1
   NuIn = NuIn';
end

CJ2 = Net.CParam.c .* (Net.CParam.J.^2) .* (1 + Net.CParam.Delta.^2);

Sigma2Out = CJ2 * (Net.SNParam.N.*NuIn) + Net.SNParam.NExt .* (Net.SNParam.JExt.^2) .* (1 + Net.SNParam.DeltaExt.^2) .* Net.SNParam.NuExt;

function NuOut = PhiVsMuSigma2(lMu, lSigma2, Net, Pop)
%
% NuOut = PhiVsMuSigma2(lMu, lSigma2, Net, Pop)
%

if size(lMu) == size(lSigma2)
else
   disp('Both lMu and lSigma2 should have the same size.');
   NuOut = [];
   return
end


for n = 1:size(lMu,1)
   for m = 1:size(lMu,2)
      NuOut(n,m) = feval(Net.SNParam.Phi{Pop}, lMu(n,m), lSigma2(n,m), ...
                         Net.SNParam.Beta(Pop), Net.SNParam.H(Pop), ...
                         Net.SNParam.Theta(Pop), Net.SNParam.Tarp(Pop));
   end
end

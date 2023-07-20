function NuOut = PhiSFA(NuIn, Net)
%
% NuOut = PhiSFA(NuIn, Net)
%

NUOUT_ERROR = 0.001;

if size(NuIn,1) == 1
   NuIn = NuIn';
end

NuOut = NuIn;
lSigma2 = Sigma2(NuIn, Net);

while 1
   NuOutPrev = NuOut;
   lMu = MuSFA(NuIn, NuOut, Net);

   for i = 1:Net.P
      NuOut(i) = feval(Net.SNParam.Phi{i}, lMu(i), lSigma2(i), Net.SNParam.Beta(i), Net.SNParam.H(i), Net.SNParam.Theta(i), Net.SNParam.Tarp(i));
   end
   
   if norm(NuOut - NuOutPrev) < NUOUT_ERROR * length(NuOut)
      break
   end
end
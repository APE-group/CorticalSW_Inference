function MuOut = MuSFA(NuIn, NuOut, Net)
%
% MuOut = MuSFA(NuIn, NuOut, Net)
%

if size(NuIn,1)==1
   NuIn = NuIn';
end

CJ = Net.CParam.c .* Net.CParam.J;

if Net.SNParam.Type == 3 % LIFCA
   MuOut = CJ * (Net.SNParam.N .* NuIn) + ...
           Net.SNParam.NExt .* Net.SNParam.JExt .* Net.SNParam.NuExt - ...
           Net.SNParam.AlphaC .* Net.SNParam.TauC .* Net.SNParam.GC .*  NuOut;
else
   MuOut = CJ * (Net.SNParam.N .* NuIn) + ...
           Net.SNParam.NExt .* Net.SNParam.JExt .* Net.SNParam.NuExt;
end
function MuOut = Mu(NuIn, Net)
%
% MuOut = Mu(NuIn, Net)
%

if size(NuIn,1)==1
   NuIn = NuIn';
end

CJ = Net.CParam.c .* Net.CParam.J;

if Net.SNParam.Type == 3 % LIFCA
   MuOut = CJ * (Net.SNParam.N .* NuIn) + ...
           Net.SNParam.NExt .* Net.SNParam.JExt .* Net.SNParam.NuExt - ...
           Net.SNParam.AlphaC .* Net.SNParam.TauC .* Net.SNParam.GC .*  NuIn + Net.SNParam.CSlow;
else
   MuOut = CJ * (Net.SNParam.N .* NuIn) + ...
           Net.SNParam.NExt .* Net.SNParam.JExt .* Net.SNParam.NuExt;
end
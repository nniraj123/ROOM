function [ind]=phasyn(phase,phi,dphi);
phmin=phase-dphi;
phmax=phase+dphi;
ind = find(phi > phmin & phi < phmax);
if phmax > 2*pi 
ind = find(phi > phmin | phi <dphi);
end
if phmin < 0 
ind = find(phi < phmax | phi > 2*pi-dphi);
end
end

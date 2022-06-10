-- elliptical
SELECT TOP 3000
g.objid, zns.nvote,zns.ra,zns.dec,
zns.p_el as elliptical,
zns.p_cw as spiralclock, zns.p_acw as spiralanticlock, zns.p_edge as edgeon,
zns.p_dk as dontknow, zns.p_mg as merger
FROM Galaxy as G
JOIN ZooNoSpec AS zns
ON G.objid = zns.objid
WHERE g.clean=1 and zns.nvote >= 30 and zns.p_el > 0.8


-- edge-on
SELECT TOP 3000
g.objid, zns.nvote,zns.ra,zns.dec,
zns.p_el as elliptical,
zns.p_cw as spiralclock, zns.p_acw as spiralanticlock, zns.p_edge as edgeon,
zns.p_dk as dontknow, zns.p_mg as merger
FROM Galaxy as G
JOIN ZooNoSpec AS zns
ON G.objid = zns.objid
WHERE g.clean=1 and zns.nvote >= 30 and zns.p_edge > 0.8

-- Spiralclock
SELECT TOP 3000
g.objid, zns.nvote,zns.ra,zns.dec,
zns.p_el as elliptical,
zns.p_cw as spiralclock, zns.p_acw as spiralanticlock, zns.p_edge as edgeon,
zns.p_dk as dontknow, zns.p_mg as merger
FROM Galaxy as G
JOIN ZooNoSpec AS zns
ON G.objid = zns.objid
WHERE g.clean=1 and zns.nvote >= 30 and zns.p_spiralclock > 0.8



-- Spiralanticlock
SELECT TOP 3000
g.objid, zns.nvote,zns.ra,zns.dec,
zns.p_el as elliptical,
zns.p_cw as spiralclock, zns.p_acw as spiralanticlock, zns.p_edge as edgeon,
zns.p_dk as dontknow, zns.p_mg as merger
FROM Galaxy as G
JOIN ZooNoSpec AS zns
ON G.objid = zns.objid
WHERE g.clean=1 and zns.nvote >= 30 and zns.p_spiralanticlock > 0.8




-- pawlikmorph
-- Round
select top 3000
m.mangaid,m.OBJRA,m.OBJDEC,m.Z,gz2_class,p.A,p.G,P.M20,P.C2080,P.C5090,P.sb0,p.reff,p.n
,m.t01_smooth_or_features_a01_smooth_weighted_fraction
,m.t07_rounded_a16_completely_round_weighted_fraction
from pawlikmorph as p
left join MaNGA_GZ2 as m on m.MANGAID = p.MANGAID
where 
--p.reff !=-99 and p.m20 !=-99 
--and p.A !=-99 and
m.t01_smooth_or_features_a01_smooth_weighted_fraction > 0.55
 and m.t07_rounded_a16_completely_round_weighted_fraction > 0.55

-- elliptical

select top 3000
m.mangaid,m.OBJRA,m.OBJDEC,m.Z,gz2_class,p.A,p.G,P.M20,P.C2080,P.C5090,P.sb0,p.reff,p.n
,m.t01_smooth_or_features_a01_smooth_weighted_fraction
,m.t07_rounded_a17_in_between_weighted_fraction
from pawlikmorph as p
left join MaNGA_GZ2 as m on m.MANGAID = p.MANGAID
where 
--p.reff !=-99 and p.m20 !=-99 
--and p.A !=-99 and
m.t01_smooth_or_features_a01_smooth_weighted_fraction > 0.6
 and m.t07_rounded_a17_in_between_weighted_fraction > 0.6


-- Edge-on
select top 3000
m.mangaid,m.OBJRA,m.OBJDEC,m.Z,gz2_class,p.A,p.G,P.M20,P.C2080,P.C5090,P.sb0,p.reff,p.n
,m.t01_smooth_or_features_a02_features_or_disk_weighted_fraction	
,m.t02_edgeon_a04_yes_weighted_fraction
from pawlikmorph as p
left join MaNGA_GZ2 as m on m.MANGAID = p.MANGAID
where 
--p.reff !=-99 and p.m20 !=-99 
--and p.A !=-99 and
m.t01_smooth_or_features_a02_features_or_disk_weighted_fraction	 > 0.55
 and m.t02_edgeon_a04_yes_weighted_fraction > 0.55
 

-- Cigar

select top 3000
m.mangaid,m.OBJRA,m.OBJDEC,m.Z,gz2_class,p.A,p.G,P.M20,P.C2080,P.C5090,P.sb0,p.reff,p.n
,m.t01_smooth_or_features_a01_smooth_weighted_fraction
,m.t07_rounded_a18_cigar_shaped_weighted_fraction
from pawlikmorph as p
left join MaNGA_GZ2 as m on m.MANGAID = p.MANGAID
where 
--p.reff !=-99 and p.m20 !=-99 
--and p.A !=-99 and
m.t01_smooth_or_features_a01_smooth_weighted_fraction > 0.55
 and m.t07_rounded_a18_cigar_shaped_weighted_fraction > 0.55      

-- SPI
select top 3000
m.mangaid,m.OBJRA,m.OBJDEC,m.Z,gz2_class,p.A,p.G,P.M20,P.C2080,P.C5090,P.sb0,p.reff,p.n
,m.t01_smooth_or_features_a02_features_or_disk_weighted_fraction	
,m.t02_edgeon_a05_no_weighted_fraction
,m.t04_spiral_a08_spiral_weighted_fraction
from pawlikmorph as p
left join MaNGA_GZ2 as m on m.MANGAID = p.MANGAID
where 
--p.reff !=-99 and p.m20 !=-99 
--and p.A !=-99 and
m.t01_smooth_or_features_a02_features_or_disk_weighted_fraction > 0.55
 and m.t02_edgeon_a05_no_weighted_fraction > 0.55
 and m.t04_spiral_a08_spiral_weighted_fraction > 0.55
 and p.G != -99

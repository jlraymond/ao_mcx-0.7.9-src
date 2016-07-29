# ao_mcx-0.7.9-src

AO-MCX: Acousto-Optic Monte Carlo eXtreme (0.7.9)

Author: Matt Adams (adamsm2@bu.edu)
Date: October, 2012 - July, 2014

Description: AO-MCX is a modified version of Qianqian Fang's Monte Carlo eXtreme (MCX)
which can be found here: mcx.sf.net. I incorporate the acousto-optic 
effect within the MCX light propagation simulation. The 'mcx' folder was originally a
"vanilla" download from the MCX website (ver 0.7.9).

Note that there are many paths specified in the make file, the source files that are
specific to the current location of the CUDA libraries on the BU engineering grid. These
need to be changed based on the system that the user compiles the code on.
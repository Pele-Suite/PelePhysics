This code enable generating usable with PelePhysics TurbInflow capabilities.
It has three modes:
 - Generation from synthetic homogeneous isotropic turbulence data,
 - Generation from a set of planes dumped from a PeleLMeX simulation using the
   PelePhysics DiagFramePlane Diagnostic capability
 - Generation from a periodic 3D plt file from a PeleLMeX simulation

See the PelePhysics documentation for more details on using this tool in all modes.

If the PeleLMeX run that produced the planes or the plt file used a mesh
mapping (`geometry.mesh_mapping`), copy that run's mapping lines verbatim
into this tool's input, e.g.

    geometry.mesh_mapping = TanhStretchMap
    TanhStretchMap.beta   = 2.0 0.0 0.0

The generator then appends a `MESHMAP_V2` trailer to `HDR` recording the map
and the precursor's computational-domain bounds, and TurbInflow samples the
file in that coordinate.  Without the trailer a file is taken to be uniform
in physical position, which is wrong for planes from a mapped run.
The example below described how to use it for synthetic turbulence data.

First generate the data using the python script:
./gen_hit_ic.py -k0 4 -N 128

To generate a synthetic HIT field discretized with 128 cells and most energetic eddies
at a wave number of 4.

Then compile the C++ executable (AMReX needed):
make

And the executable to generate the turbfile (adapt the input file to your needs):
./PeleTurb3d.gnu.ex input hit_file=hit_ic_4_128.dat input_ncell=128

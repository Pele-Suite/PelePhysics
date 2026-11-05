#include "main.H"

using namespace amrex;

static void
print_usage(int, char* argv[])
{
  std::cerr << "usage:\n";
  std::cerr << argv[0]
            << " type=<turb_box, periodic_plt, diag_frame_planes>"
               " ofile=<output directory name>\n";
  std::cerr << "   Optional for type=diag_frame_planes and periodic_plt:\n";
  std::cerr << "      scalars=<plt variables to store next to the velocity,"
               " e.g. 'temp Y(N2)'>\n";
  std::cerr << "   Additional inputs for type=diag_frame_planes:\n";
  std::cerr << "      ifiles=<list of DiagFramePlanes to include>\n";
  std::cerr << "      level=<AMR level from DiagFramePlanes to use, DEF=0>\n";
  std::cerr << "      periodicity=<of input planes, DEF='1 1 1'>\n";
  std::cerr << "      normal=<normal dir for input DiagFramePlanes>\n";
  std::cerr << "   Additional inputs for type=periodic_plt:\n";
  std::cerr << "      ifile=<Plt file, must be per in norm direction>\n";
  std::cerr << "      normal=<normal direction for planes being extracted>\n";
  std::cerr << "      level=<AMR level from Plot file to use, DEF=0>\n";
  std::cerr << "      periodicity=<periodicity of Plot file, DEF='1 1 1'>\n";
  std::cerr << "   Additional inputs for type=turb_box:\n";
  std::cerr << "      hit_file=<cubic isotropic turbulence file to use>\n";
  std::cerr << "      input_ncell=<ncells in each direction in turb_file>\n";
  std::cerr << "      input_binaryformat=<infile is binary not ASCII>\n";
  std::cerr << "      urms0=<desired velocity fluctuation scale>\n";
  exit(1);
}

int
main(int argc, char* argv[])
{
  Initialize(argc, argv);
  {
    if (argc < 2)
      print_usage(argc, argv);

    ParmParse pp;

    const std::string farg = amrex::get_command_argument(1);
    if (farg == "-h" || farg == "--help")
      print_usage(argc, argv);

    std::string ofile;
    pp.get("ofile", ofile);
    std::string TurbDir = ofile;
    if (ParallelDescriptor::IOProcessor())
      if (!UtilCreateDirectory(TurbDir, 0755))
        CreateDirectoryFailed(TurbDir);

    std::string Hdr = TurbDir;
    Hdr += "/HDR";
    std::string Dat = TurbDir;
    Dat += "/DAT";

    std::ofstream ifsd, ifsh;

    ifsh.open(Hdr.c_str(), std::ios::out | std::ios::trunc);
    if (!ifsh.good())
      FileOpenFailed(Hdr);

    ifsd.open(Dat.c_str(), std::ios::out | std::ios::trunc);
    if (!ifsd.good())
      FileOpenFailed(Dat);

    InputType itype;
    pp.get("type", itype);
    if (itype == InputType::diag_frame_planes) {
      turbinflow_from_diag_frame_planes(ifsd, ifsh);
    } else if (itype == InputType::periodic_plt) {
      turbinflow_from_periodic_plt(ifsd, ifsh);
    } else {
      turbinflow_from_turb_box(ifsd, ifsh);
    }
  }
  Finalize();
}

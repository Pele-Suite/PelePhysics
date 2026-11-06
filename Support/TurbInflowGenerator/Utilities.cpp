#include "main.H"

using namespace amrex;

AMREX_FORCE_INLINE
std::string
read_file(std::ifstream& in)
{
  return static_cast<std::stringstream const&>(
           std::stringstream() << in.rdbuf())
    .str();
}

void
write_meshmap_trailer(
  std::ofstream& ifsh, const RealBox& probDomain, const IntVect& dim_map)
{
  using pele::physics::MeshMapEvaluator;
  const MeshMapEvaluator e =
    pele::physics::make_evaluator_from_inputs(probDomain);
  if (e.m_kind == MeshMapEvaluator::Kind::Identity) {
    return;
  }

  // A stretched normal direction cannot be expressed in the file (the
  // normal is time for diag_frame_planes, and a uniformly convected
  // coordinate for periodic_plt).  Warn; it is the user's precursor.
  const int dn = dim_map[2];
  const bool normal_stretched =
    (e.m_kind == MeshMapEvaluator::Kind::Constant)
      ? (std::abs(e.m_p[dn] - Real(1.0)) > MeshMapEvaluator::BETA_EPS)
      : (std::abs(e.m_p[dn]) > MeshMapEvaluator::BETA_EPS ||
         std::abs(e.m_p2[dn]) > MeshMapEvaluator::BETA_EPS);
  if (normal_stretched) {
    amrex::Print()
      << "WARNING: geometry.mesh_mapping stretches the file's normal "
         "direction (plotfile axis "
      << dn
      << "); the turbulence file cannot represent that and TurbInflow "
         "will treat the normal as uniform.\n";
  }

  const auto prec = ifsh.precision();
  ifsh << std::setprecision(17);
  ifsh << "MESHMAP_V2\n";
  for (int t = 0; t < 2; ++t) {
    const int d = dim_map[t];
    ifsh << static_cast<int>(e.m_kind) << ' ' << e.m_p[d] << ' ' << e.m_p2[d]
         << ' ' << e.m_p3[d] << ' ' << e.m_q[d] << ' ' << probDomain.lo(d)
         << ' ' << probDomain.hi(d) << '\n';
  }
  ifsh << std::setprecision(prec);
  amrex::Print() << "Wrote MESHMAP_V2 trailer (map kind "
                 << static_cast<int>(e.m_kind)
                 << "): the file is uniform in the precursor's Xi "
                    "coordinate\n";
}

// Map from save domain (z-normal, x,y-tangential) to input domain (arbitrary)
IntVect
index_mapper(
  const IntVect& idx_in,
  const IntVect& dim_map,
  const Box& domain,
  const IntVect& periodic)
{
  IntVect idx_out;
  // periodic directions - populate periodically
  // non periodic directions - FOExtrap
  // new x and y are shifted up/right by 1 cell
  // new z-direction must be a single cell width
  for (int idim = 0; idim < 2; idim++) {
    idx_out[dim_map[idim]] =
      periodic[dim_map[idim]]
        ? domain.smallEnd(dim_map[idim]) +
            (domain.length(dim_map[idim]) + idx_in[idim] - 1) %
              domain.length(dim_map[idim])
        : amrex::Clamp(
            idx_in[idim] - 1, domain.smallEnd(dim_map[idim]),
            domain.bigEnd(dim_map[idim]));
  }
  idx_out[dim_map[2]] = idx_in[2];
  return idx_out;
}

// -----------------------------------------------------------
// Read a binary file
// INPUTS/OUTPUTS:
// iname => filename
// nx    => input resolution
// ny    => input resolution
// nz    => input resolution
// data  <= output data
// -----------------------------------------------------------
void
read_binary(
  const std::string& iname,
  const size_t nx,
  const size_t ny,
  const size_t nz,
  const size_t ncol,
  amrex::Vector<double>& data /*needs to be double*/)
{
  std::ifstream infile(iname, std::ios::in | std::ios::binary);
  if (not infile.is_open()) {
    amrex::Abort("Unable to open input file " + iname);
  }

  for (size_t i = 0; i < nx * ny * nz * ncol; i++) {
    infile.read(reinterpret_cast<char*>(&data[i]), sizeof(data[i]));
  }
  infile.close();
}

// -----------------------------------------------------------
// Read a csv file
// INPUTS/OUTPUTS:
// iname => filename
// nx    => input resolution
// ny    => input resolution
// nz    => input resolution
// data  <= output data
// -----------------------------------------------------------
void
read_csv(
  const std::string& iname,
  const size_t nx,
  const size_t ny,
  const size_t nz,
  amrex::Vector<amrex::Real>& data)
{
  std::ifstream infile(iname, std::ios::in);
  const std::string memfile = read_file(infile);
  if (not infile.is_open()) {
    amrex::Abort("Unable to open input file " + iname);
  }
  infile.close();
  std::istringstream iss(memfile);

  // Read the file
  size_t nlines = 0;
  std::string firstline;
  std::string line;
  std::getline(iss, firstline); // skip header
  while (getline(iss, line)) {
    ++nlines;
  }

  // Quick sanity check
  if (nlines != nx * ny * nz) {
    amrex::Abort(
      "Number of lines in the input file (= " + std::to_string(nlines) +
      ") does not match the input resolution (=" + std::to_string(nx) + ")");
  }

  // Read the data from the file
  iss.clear();
  iss.seekg(0, std::ios::beg);
  std::getline(iss, firstline); // skip header
  int cnt = 0;
  while (std::getline(iss, line)) {
    std::istringstream linestream(line);
    std::string value;
    while (getline(linestream, value, ',')) {
      std::istringstream sinput(value);
      sinput >> data[cnt];
      cnt++;
    }
  }
}

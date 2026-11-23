#ifndef PELE_TESTCHECK_H
#define PELE_TESTCHECK_H

// PASS/FAIL checks shared by the standalone tests (Soot, Spray, Radiation)

#include <AMReX_Extension.H>
#include <AMReX_GpuQualifiers.H>
#include <AMReX_Math.H>
#include <AMReX_Print.H>
#include <AMReX_REAL.H>

#include <string>

namespace pele::testing {

// Prints a line per check and counts the outcomes. A check tagged with the
// fix it needs (e.g. "#732") is an expected failure: failing prints XFAIL and
// is not an error; passing prints XPASS and is an error, so that the fix
// removes the tag.
class Checker
{
public:
  void operator()(
    const bool ok, const std::string& what, const std::string& tag = "")
  {
    if (tag.empty()) {
      amrex::Print() << (ok ? "  PASS: " : "  FAIL: ") << what << "\n";
      ++(ok ? m_npass : m_nfail);
    } else if (ok) {
      amrex::Print() << "  XPASS (" << tag << "): " << what
                     << " - remove the tag\n";
      ++m_nxpass;
    } else {
      amrex::Print() << "  XFAIL (" << tag << "): " << what << "\n";
      ++m_nxfail;
    }
  }

  // Number of errors: FAIL and XPASS
  int errors() const { return m_nfail + m_nxpass; }

  // Prints the totals and returns the exit code, non-zero on any error
  int summary(const std::string& label) const
  {
    amrex::Print() << "\n"
                   << label << " checks: " << m_npass << " PASS, " << m_nxfail
                   << " XFAIL, " << m_nfail << " FAIL, " << m_nxpass
                   << " XPASS\n";
    return (errors() > 0) ? 1 : 0;
  }

private:
  int m_npass = 0;
  int m_nfail = 0;
  int m_nxfail = 0;
  int m_nxpass = 0;
};

// |a - b| / |b|, or |a| when b is zero
AMREX_GPU_HOST_DEVICE AMREX_FORCE_INLINE amrex::Real
relErr(const amrex::Real a, const amrex::Real b)
{
  return (b == 0.) ? amrex::Math::abs(a)
                   : amrex::Math::abs(a - b) / amrex::Math::abs(b);
}

} // namespace pele::testing

#endif

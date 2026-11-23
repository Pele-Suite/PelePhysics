#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <AMReX_Array.H>
#include <iostream>
#include <utility>

namespace PeleRad {

struct RadComps
{
  int co2Indx = -1;
  int h2oIndx = -1;
  int coIndx = -1;
  int ch4Indx = -1;
  int c2h4Indx = -1;

  void checkIndices() const
  {
    amrex::Array<std::pair<char const*, int>, 5> const species{
      {{"CO2", co2Indx},
       {"H2O", h2oIndx},
       {"CO", coIndx},
       {"CH4", ch4Indx},
       {"C2H4", c2h4Indx}}};
    for (auto const& [name, indx] : species) {
      if (indx >= 0) {
        std::cout << " include radiative gas: " << name << ", the index is "
                  << indx << std::endl;
      }
    }
  }
};

} // namespace PeleRad
#endif

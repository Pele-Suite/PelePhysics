#ifndef CONSTANTS_H
#define CONSTANTS_H

namespace PeleRad {

struct RadComps
{
  int co2Indx = -1;
  int h2oIndx = -1;
  int coIndx = -1;
  int ch4Indx = -1;
  int c2h4Indx = -1;

  void checkIndices()
  {
    if (co2Indx >= 0) {
      std::cout << " include radiative gas: CO2, the index is " << co2Indx
                << std::endl;
    }
    if (h2oIndx >= 0) {
      std::cout << " include radiative gas: H2O, the index is " << h2oIndx
                << std::endl;
    }
    if (coIndx >= 0) {
      std::cout << " include radiative gas: CO, the index is " << coIndx
                << std::endl;
    }
    if (ch4Indx >= 0) {
      std::cout << " include radiative gas: CH4, the index is " << ch4Indx
                << std::endl;
    }
    if (c2h4Indx >= 0) {
      std::cout << " include radiative gas: C2H4, the index is " << c2h4Indx
                << std::endl;
    }
  }
};

} // namespace PeleRad
#endif

#include "DiagIsosurface.H"

#include <AMReX_MultiFabUtil.H>
#include <AMReX_iMultiFab.H>
#include <AMReX_FArrayBox.H>
#include <AMReX_Loop.H>

#include <algorithm>
#include <array>
#include <cmath>
#include <fstream>
#include <functional>
#include <iomanip>
#include <limits>
#include <unordered_map>

namespace {

constexpr int nodesPerElt = AMREX_SPACEDIM;

// A dual-grid edge joins two cells differing by one along a single axis: it
// is identified by its level, its low cell and that axis.
constexpr int keySize = AMREX_SPACEDIM + 2;
using EdgeKey = std::array<int, keySize>;

struct EdgeKeyHash
{
  std::size_t operator()(const EdgeKey& a_key) const noexcept
  {
    std::size_t h = 0;
    for (const int v : a_key) {
      h ^= std::hash<int>{}(v) + 0x9e3779b9 + (h << 6) + (h >> 2);
    }
    return h;
  }
};

using EdgeMap = std::unordered_map<EdgeKey, int, EdgeKeyHash>;

// Rank-local surface under construction
struct LocalSurface
{
  std::vector<amrex::Real> nodes; // nNodeComp values per node
  std::vector<int> keys;          // EdgeKey per node, flattened
  std::vector<int> elts;          // AMREX_SPACEDIM node indices per element
  EdgeMap edgeToNode;
  amrex::Real measure{0.0};
};

// Corner flags built on the device for the marching
constexpr int usableBit = 1; // corner can be used by a cube
constexpr int levelBit = 2;  // corner is a cell of the level (valid somewhere)

// Tolerance used when snapping an edge crossing onto a corner, as in
// PeleAnalysis' isosurface tool.
constexpr amrex::Real isoEps = 1.0e-15;

#if AMREX_SPACEDIM == 2
// Marching squares. Corners: 0=(0,0) 1=(1,0) 2=(1,1) 3=(0,1).
constexpr int nCorners = 4;
constexpr int cornerOffset[nCorners][2] = {{0, 0}, {1, 0}, {1, 1}, {0, 1}};
constexpr int nEdges = 4;
constexpr int edgeCorners[nEdges][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0}};
// Up to two segments per square, as pairs of edge indices.
constexpr int segTable[16][4] = {
  {-1, -1, -1, -1}, {0, 3, -1, -1}, {0, 1, -1, -1}, {1, 3, -1, -1},
  {1, 2, -1, -1},   {0, 1, 2, 3},   {0, 2, -1, -1}, {2, 3, -1, -1},
  {2, 3, -1, -1},   {0, 2, -1, -1}, {0, 1, 2, 3},   {1, 2, -1, -1},
  {1, 3, -1, -1},   {0, 1, -1, -1}, {0, 3, -1, -1}, {-1, -1, -1, -1}};
#else
// Marching cubes (Lorensen & Cline; tables from Paul Bourke). Corners:
// 0=(0,0,0) 1=(1,0,0) 2=(1,1,0) 3=(0,1,0) 4=(0,0,1) 5=(1,0,1) 6=(1,1,1)
// 7=(0,1,1).
constexpr int nCorners = 8;
constexpr int cornerOffset[nCorners][3] = {{0, 0, 0}, {1, 0, 0}, {1, 1, 0},
                                           {0, 1, 0}, {0, 0, 1}, {1, 0, 1},
                                           {1, 1, 1}, {0, 1, 1}};
constexpr int nEdges = 12;
constexpr int edgeCorners[nEdges][2] = {{0, 1}, {1, 2}, {2, 3}, {3, 0},
                                        {4, 5}, {5, 6}, {6, 7}, {7, 4},
                                        {0, 4}, {1, 5}, {2, 6}, {3, 7}};

// clang-format off
constexpr int edgeTable[256] = {
  0x0, 0x109, 0x203, 0x30a, 0x406, 0x50f, 0x605, 0x70c,
  0x80c, 0x905, 0xa0f, 0xb06, 0xc0a, 0xd03, 0xe09, 0xf00,
  0x190, 0x99, 0x393, 0x29a, 0x596, 0x49f, 0x795, 0x69c,
  0x99c, 0x895, 0xb9f, 0xa96, 0xd9a, 0xc93, 0xf99, 0xe90,
  0x230, 0x339, 0x33, 0x13a, 0x636, 0x73f, 0x435, 0x53c,
  0xa3c, 0xb35, 0x83f, 0x936, 0xe3a, 0xf33, 0xc39, 0xd30,
  0x3a0, 0x2a9, 0x1a3, 0xaa, 0x7a6, 0x6af, 0x5a5, 0x4ac,
  0xbac, 0xaa5, 0x9af, 0x8a6, 0xfaa, 0xea3, 0xda9, 0xca0,
  0x460, 0x569, 0x663, 0x76a, 0x66, 0x16f, 0x265, 0x36c,
  0xc6c, 0xd65, 0xe6f, 0xf66, 0x86a, 0x963, 0xa69, 0xb60,
  0x5f0, 0x4f9, 0x7f3, 0x6fa, 0x1f6, 0xff, 0x3f5, 0x2fc,
  0xdfc, 0xcf5, 0xfff, 0xef6, 0x9fa, 0x8f3, 0xbf9, 0xaf0,
  0x650, 0x759, 0x453, 0x55a, 0x256, 0x35f, 0x55, 0x15c,
  0xe5c, 0xf55, 0xc5f, 0xd56, 0xa5a, 0xb53, 0x859, 0x950,
  0x7c0, 0x6c9, 0x5c3, 0x4ca, 0x3c6, 0x2cf, 0x1c5, 0xcc,
  0xfcc, 0xec5, 0xdcf, 0xcc6, 0xbca, 0xac3, 0x9c9, 0x8c0,
  0x8c0, 0x9c9, 0xac3, 0xbca, 0xcc6, 0xdcf, 0xec5, 0xfcc,
  0xcc, 0x1c5, 0x2cf, 0x3c6, 0x4ca, 0x5c3, 0x6c9, 0x7c0,
  0x950, 0x859, 0xb53, 0xa5a, 0xd56, 0xc5f, 0xf55, 0xe5c,
  0x15c, 0x55, 0x35f, 0x256, 0x55a, 0x453, 0x759, 0x650,
  0xaf0, 0xbf9, 0x8f3, 0x9fa, 0xef6, 0xfff, 0xcf5, 0xdfc,
  0x2fc, 0x3f5, 0xff, 0x1f6, 0x6fa, 0x7f3, 0x4f9, 0x5f0,
  0xb60, 0xa69, 0x963, 0x86a, 0xf66, 0xe6f, 0xd65, 0xc6c,
  0x36c, 0x265, 0x16f, 0x66, 0x76a, 0x663, 0x569, 0x460,
  0xca0, 0xda9, 0xea3, 0xfaa, 0x8a6, 0x9af, 0xaa5, 0xbac,
  0x4ac, 0x5a5, 0x6af, 0x7a6, 0xaa, 0x1a3, 0x2a9, 0x3a0,
  0xd30, 0xc39, 0xf33, 0xe3a, 0x936, 0x83f, 0xb35, 0xa3c,
  0x53c, 0x435, 0x73f, 0x636, 0x13a, 0x33, 0x339, 0x230,
  0xe90, 0xf99, 0xc93, 0xd9a, 0xa96, 0xb9f, 0x895, 0x99c,
  0x69c, 0x795, 0x49f, 0x596, 0x29a, 0x393, 0x99, 0x190,
  0xf00, 0xe09, 0xd03, 0xc0a, 0xb06, 0xa0f, 0x905, 0x80c,
  0x70c, 0x605, 0x50f, 0x406, 0x30a, 0x203, 0x109, 0x0,
};

constexpr int triTable[256][16] = {
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 8, 3, 9, 8, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 3, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 2, 10, 0, 2, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 8, 3, 2, 10, 8, 10, 9, 8, -1, -1, -1, -1, -1, -1, -1},
  {3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 11, 2, 8, 11, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 9, 0, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 11, 2, 1, 9, 11, 9, 8, 11, -1, -1, -1, -1, -1, -1, -1},
  {3, 10, 1, 11, 10, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 10, 1, 0, 8, 10, 8, 11, 10, -1, -1, -1, -1, -1, -1, -1},
  {3, 9, 0, 3, 11, 9, 11, 10, 9, -1, -1, -1, -1, -1, -1, -1},
  {9, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 3, 0, 7, 3, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 9, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 1, 9, 4, 7, 1, 7, 3, 1, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 10, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 4, 7, 3, 0, 4, 1, 2, 10, -1, -1, -1, -1, -1, -1, -1},
  {9, 2, 10, 9, 0, 2, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
  {2, 10, 9, 2, 9, 7, 2, 7, 3, 7, 9, 4, -1, -1, -1, -1},
  {8, 4, 7, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {11, 4, 7, 11, 2, 4, 2, 0, 4, -1, -1, -1, -1, -1, -1, -1},
  {9, 0, 1, 8, 4, 7, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
  {4, 7, 11, 9, 4, 11, 9, 11, 2, 9, 2, 1, -1, -1, -1, -1},
  {3, 10, 1, 3, 11, 10, 7, 8, 4, -1, -1, -1, -1, -1, -1, -1},
  {1, 11, 10, 1, 4, 11, 1, 0, 4, 7, 11, 4, -1, -1, -1, -1},
  {4, 7, 8, 9, 0, 11, 9, 11, 10, 11, 0, 3, -1, -1, -1, -1},
  {4, 7, 11, 4, 11, 9, 9, 11, 10, -1, -1, -1, -1, -1, -1, -1},
  {9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 5, 4, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 5, 4, 1, 5, 0, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 5, 4, 8, 3, 5, 3, 1, 5, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 10, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 0, 8, 1, 2, 10, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
  {5, 2, 10, 5, 4, 2, 4, 0, 2, -1, -1, -1, -1, -1, -1, -1},
  {2, 10, 5, 3, 2, 5, 3, 5, 4, 3, 4, 8, -1, -1, -1, -1},
  {9, 5, 4, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 11, 2, 0, 8, 11, 4, 9, 5, -1, -1, -1, -1, -1, -1, -1},
  {0, 5, 4, 0, 1, 5, 2, 3, 11, -1, -1, -1, -1, -1, -1, -1},
  {2, 1, 5, 2, 5, 8, 2, 8, 11, 4, 8, 5, -1, -1, -1, -1},
  {10, 3, 11, 10, 1, 3, 9, 5, 4, -1, -1, -1, -1, -1, -1, -1},
  {4, 9, 5, 0, 8, 1, 8, 10, 1, 8, 11, 10, -1, -1, -1, -1},
  {5, 4, 0, 5, 0, 11, 5, 11, 10, 11, 0, 3, -1, -1, -1, -1},
  {5, 4, 8, 5, 8, 10, 10, 8, 11, -1, -1, -1, -1, -1, -1, -1},
  {9, 7, 8, 5, 7, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 3, 0, 9, 5, 3, 5, 7, 3, -1, -1, -1, -1, -1, -1, -1},
  {0, 7, 8, 0, 1, 7, 1, 5, 7, -1, -1, -1, -1, -1, -1, -1},
  {1, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 7, 8, 9, 5, 7, 10, 1, 2, -1, -1, -1, -1, -1, -1, -1},
  {10, 1, 2, 9, 5, 0, 5, 3, 0, 5, 7, 3, -1, -1, -1, -1},
  {8, 0, 2, 8, 2, 5, 8, 5, 7, 10, 5, 2, -1, -1, -1, -1},
  {2, 10, 5, 2, 5, 3, 3, 5, 7, -1, -1, -1, -1, -1, -1, -1},
  {7, 9, 5, 7, 8, 9, 3, 11, 2, -1, -1, -1, -1, -1, -1, -1},
  {9, 5, 7, 9, 7, 2, 9, 2, 0, 2, 7, 11, -1, -1, -1, -1},
  {2, 3, 11, 0, 1, 8, 1, 7, 8, 1, 5, 7, -1, -1, -1, -1},
  {11, 2, 1, 11, 1, 7, 7, 1, 5, -1, -1, -1, -1, -1, -1, -1},
  {9, 5, 8, 8, 5, 7, 10, 1, 3, 10, 3, 11, -1, -1, -1, -1},
  {5, 7, 0, 5, 0, 9, 7, 11, 0, 1, 0, 10, 11, 10, 0, -1},
  {11, 10, 0, 11, 0, 3, 10, 5, 0, 8, 0, 7, 5, 7, 0, -1},
  {11, 10, 5, 7, 11, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 3, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 0, 1, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 8, 3, 1, 9, 8, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
  {1, 6, 5, 2, 6, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 6, 5, 1, 2, 6, 3, 0, 8, -1, -1, -1, -1, -1, -1, -1},
  {9, 6, 5, 9, 0, 6, 0, 2, 6, -1, -1, -1, -1, -1, -1, -1},
  {5, 9, 8, 5, 8, 2, 5, 2, 6, 3, 2, 8, -1, -1, -1, -1},
  {2, 3, 11, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {11, 0, 8, 11, 2, 0, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 9, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1, -1, -1, -1},
  {5, 10, 6, 1, 9, 2, 9, 11, 2, 9, 8, 11, -1, -1, -1, -1},
  {6, 3, 11, 6, 5, 3, 5, 1, 3, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 11, 0, 11, 5, 0, 5, 1, 5, 11, 6, -1, -1, -1, -1},
  {3, 11, 6, 0, 3, 6, 0, 6, 5, 0, 5, 9, -1, -1, -1, -1},
  {6, 5, 9, 6, 9, 11, 11, 9, 8, -1, -1, -1, -1, -1, -1, -1},
  {5, 10, 6, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 3, 0, 4, 7, 3, 6, 5, 10, -1, -1, -1, -1, -1, -1, -1},
  {1, 9, 0, 5, 10, 6, 8, 4, 7, -1, -1, -1, -1, -1, -1, -1},
  {10, 6, 5, 1, 9, 7, 1, 7, 3, 7, 9, 4, -1, -1, -1, -1},
  {6, 1, 2, 6, 5, 1, 4, 7, 8, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 5, 5, 2, 6, 3, 0, 4, 3, 4, 7, -1, -1, -1, -1},
  {8, 4, 7, 9, 0, 5, 0, 6, 5, 0, 2, 6, -1, -1, -1, -1},
  {7, 3, 9, 7, 9, 4, 3, 2, 9, 5, 9, 6, 2, 6, 9, -1},
  {3, 11, 2, 7, 8, 4, 10, 6, 5, -1, -1, -1, -1, -1, -1, -1},
  {5, 10, 6, 4, 7, 2, 4, 2, 0, 2, 7, 11, -1, -1, -1, -1},
  {0, 1, 9, 4, 7, 8, 2, 3, 11, 5, 10, 6, -1, -1, -1, -1},
  {9, 2, 1, 9, 11, 2, 9, 4, 11, 7, 11, 4, 5, 10, 6, -1},
  {8, 4, 7, 3, 11, 5, 3, 5, 1, 5, 11, 6, -1, -1, -1, -1},
  {5, 1, 11, 5, 11, 6, 1, 0, 11, 7, 11, 4, 0, 4, 11, -1},
  {0, 5, 9, 0, 6, 5, 0, 3, 6, 11, 6, 3, 8, 4, 7, -1},
  {6, 5, 9, 6, 9, 11, 4, 7, 9, 7, 11, 9, -1, -1, -1, -1},
  {10, 4, 9, 6, 4, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 10, 6, 4, 9, 10, 0, 8, 3, -1, -1, -1, -1, -1, -1, -1},
  {10, 0, 1, 10, 6, 0, 6, 4, 0, -1, -1, -1, -1, -1, -1, -1},
  {8, 3, 1, 8, 1, 6, 8, 6, 4, 6, 1, 10, -1, -1, -1, -1},
  {1, 4, 9, 1, 2, 4, 2, 6, 4, -1, -1, -1, -1, -1, -1, -1},
  {3, 0, 8, 1, 2, 9, 2, 4, 9, 2, 6, 4, -1, -1, -1, -1},
  {0, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 3, 2, 8, 2, 4, 4, 2, 6, -1, -1, -1, -1, -1, -1, -1},
  {10, 4, 9, 10, 6, 4, 11, 2, 3, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 2, 2, 8, 11, 4, 9, 10, 4, 10, 6, -1, -1, -1, -1},
  {3, 11, 2, 0, 1, 6, 0, 6, 4, 6, 1, 10, -1, -1, -1, -1},
  {6, 4, 1, 6, 1, 10, 4, 8, 1, 2, 1, 11, 8, 11, 1, -1},
  {9, 6, 4, 9, 3, 6, 9, 1, 3, 11, 6, 3, -1, -1, -1, -1},
  {8, 11, 1, 8, 1, 0, 11, 6, 1, 9, 1, 4, 6, 4, 1, -1},
  {3, 11, 6, 3, 6, 0, 0, 6, 4, -1, -1, -1, -1, -1, -1, -1},
  {6, 4, 8, 11, 6, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {7, 10, 6, 7, 8, 10, 8, 9, 10, -1, -1, -1, -1, -1, -1, -1},
  {0, 7, 3, 0, 10, 7, 0, 9, 10, 6, 7, 10, -1, -1, -1, -1},
  {10, 6, 7, 1, 10, 7, 1, 7, 8, 1, 8, 0, -1, -1, -1, -1},
  {10, 6, 7, 10, 7, 1, 1, 7, 3, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 6, 1, 6, 8, 1, 8, 9, 8, 6, 7, -1, -1, -1, -1},
  {2, 6, 9, 2, 9, 1, 6, 7, 9, 0, 9, 3, 7, 3, 9, -1},
  {7, 8, 0, 7, 0, 6, 6, 0, 2, -1, -1, -1, -1, -1, -1, -1},
  {7, 3, 2, 6, 7, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 3, 11, 10, 6, 8, 10, 8, 9, 8, 6, 7, -1, -1, -1, -1},
  {2, 0, 7, 2, 7, 11, 0, 9, 7, 6, 7, 10, 9, 10, 7, -1},
  {1, 8, 0, 1, 7, 8, 1, 10, 7, 6, 7, 10, 2, 3, 11, -1},
  {11, 2, 1, 11, 1, 7, 10, 6, 1, 6, 7, 1, -1, -1, -1, -1},
  {8, 9, 6, 8, 6, 7, 9, 1, 6, 11, 6, 3, 1, 3, 6, -1},
  {0, 9, 1, 11, 6, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {7, 8, 0, 7, 0, 6, 3, 11, 0, 11, 6, 0, -1, -1, -1, -1},
  {7, 11, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 0, 8, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 9, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 1, 9, 8, 3, 1, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
  {10, 1, 2, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 10, 3, 0, 8, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
  {2, 9, 0, 2, 10, 9, 6, 11, 7, -1, -1, -1, -1, -1, -1, -1},
  {6, 11, 7, 2, 10, 3, 10, 8, 3, 10, 9, 8, -1, -1, -1, -1},
  {7, 2, 3, 6, 2, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {7, 0, 8, 7, 6, 0, 6, 2, 0, -1, -1, -1, -1, -1, -1, -1},
  {2, 7, 6, 2, 3, 7, 0, 1, 9, -1, -1, -1, -1, -1, -1, -1},
  {1, 6, 2, 1, 8, 6, 1, 9, 8, 8, 7, 6, -1, -1, -1, -1},
  {10, 7, 6, 10, 1, 7, 1, 3, 7, -1, -1, -1, -1, -1, -1, -1},
  {10, 7, 6, 1, 7, 10, 1, 8, 7, 1, 0, 8, -1, -1, -1, -1},
  {0, 3, 7, 0, 7, 10, 0, 10, 9, 6, 10, 7, -1, -1, -1, -1},
  {7, 6, 10, 7, 10, 8, 8, 10, 9, -1, -1, -1, -1, -1, -1, -1},
  {6, 8, 4, 11, 8, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 6, 11, 3, 0, 6, 0, 4, 6, -1, -1, -1, -1, -1, -1, -1},
  {8, 6, 11, 8, 4, 6, 9, 0, 1, -1, -1, -1, -1, -1, -1, -1},
  {9, 4, 6, 9, 6, 3, 9, 3, 1, 11, 3, 6, -1, -1, -1, -1},
  {6, 8, 4, 6, 11, 8, 2, 10, 1, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 10, 3, 0, 11, 0, 6, 11, 0, 4, 6, -1, -1, -1, -1},
  {4, 11, 8, 4, 6, 11, 0, 2, 9, 2, 10, 9, -1, -1, -1, -1},
  {10, 9, 3, 10, 3, 2, 9, 4, 3, 11, 3, 6, 4, 6, 3, -1},
  {8, 2, 3, 8, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1},
  {0, 4, 2, 4, 6, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 9, 0, 2, 3, 4, 2, 4, 6, 4, 3, 8, -1, -1, -1, -1},
  {1, 9, 4, 1, 4, 2, 2, 4, 6, -1, -1, -1, -1, -1, -1, -1},
  {8, 1, 3, 8, 6, 1, 8, 4, 6, 6, 10, 1, -1, -1, -1, -1},
  {10, 1, 0, 10, 0, 6, 6, 0, 4, -1, -1, -1, -1, -1, -1, -1},
  {4, 6, 3, 4, 3, 8, 6, 10, 3, 0, 3, 9, 10, 9, 3, -1},
  {10, 9, 4, 6, 10, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 9, 5, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 3, 4, 9, 5, 11, 7, 6, -1, -1, -1, -1, -1, -1, -1},
  {5, 0, 1, 5, 4, 0, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
  {11, 7, 6, 8, 3, 4, 3, 5, 4, 3, 1, 5, -1, -1, -1, -1},
  {9, 5, 4, 10, 1, 2, 7, 6, 11, -1, -1, -1, -1, -1, -1, -1},
  {6, 11, 7, 1, 2, 10, 0, 8, 3, 4, 9, 5, -1, -1, -1, -1},
  {7, 6, 11, 5, 4, 10, 4, 2, 10, 4, 0, 2, -1, -1, -1, -1},
  {3, 4, 8, 3, 5, 4, 3, 2, 5, 10, 5, 2, 11, 7, 6, -1},
  {7, 2, 3, 7, 6, 2, 5, 4, 9, -1, -1, -1, -1, -1, -1, -1},
  {9, 5, 4, 0, 8, 6, 0, 6, 2, 6, 8, 7, -1, -1, -1, -1},
  {3, 6, 2, 3, 7, 6, 1, 5, 0, 5, 4, 0, -1, -1, -1, -1},
  {6, 2, 8, 6, 8, 7, 2, 1, 8, 4, 8, 5, 1, 5, 8, -1},
  {9, 5, 4, 10, 1, 6, 1, 7, 6, 1, 3, 7, -1, -1, -1, -1},
  {1, 6, 10, 1, 7, 6, 1, 0, 7, 8, 7, 0, 9, 5, 4, -1},
  {4, 0, 10, 4, 10, 5, 0, 3, 10, 6, 10, 7, 3, 7, 10, -1},
  {7, 6, 10, 7, 10, 8, 5, 4, 10, 4, 8, 10, -1, -1, -1, -1},
  {6, 9, 5, 6, 11, 9, 11, 8, 9, -1, -1, -1, -1, -1, -1, -1},
  {3, 6, 11, 0, 6, 3, 0, 5, 6, 0, 9, 5, -1, -1, -1, -1},
  {0, 11, 8, 0, 5, 11, 0, 1, 5, 5, 6, 11, -1, -1, -1, -1},
  {6, 11, 3, 6, 3, 5, 5, 3, 1, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 10, 9, 5, 11, 9, 11, 8, 11, 5, 6, -1, -1, -1, -1},
  {0, 11, 3, 0, 6, 11, 0, 9, 6, 5, 6, 9, 1, 2, 10, -1},
  {11, 8, 5, 11, 5, 6, 8, 0, 5, 10, 5, 2, 0, 2, 5, -1},
  {6, 11, 3, 6, 3, 5, 2, 10, 3, 10, 5, 3, -1, -1, -1, -1},
  {5, 8, 9, 5, 2, 8, 5, 6, 2, 3, 8, 2, -1, -1, -1, -1},
  {9, 5, 6, 9, 6, 0, 0, 6, 2, -1, -1, -1, -1, -1, -1, -1},
  {1, 5, 8, 1, 8, 0, 5, 6, 8, 3, 8, 2, 6, 2, 8, -1},
  {1, 5, 6, 2, 1, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 3, 6, 1, 6, 10, 3, 8, 6, 5, 6, 9, 8, 9, 6, -1},
  {10, 1, 0, 10, 0, 6, 9, 5, 0, 5, 6, 0, -1, -1, -1, -1},
  {0, 3, 8, 5, 6, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {10, 5, 6, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {11, 5, 10, 7, 5, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {11, 5, 10, 11, 7, 5, 8, 3, 0, -1, -1, -1, -1, -1, -1, -1},
  {5, 11, 7, 5, 10, 11, 1, 9, 0, -1, -1, -1, -1, -1, -1, -1},
  {10, 7, 5, 10, 11, 7, 9, 8, 1, 8, 3, 1, -1, -1, -1, -1},
  {11, 1, 2, 11, 7, 1, 7, 5, 1, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 3, 1, 2, 7, 1, 7, 5, 7, 2, 11, -1, -1, -1, -1},
  {9, 7, 5, 9, 2, 7, 9, 0, 2, 2, 11, 7, -1, -1, -1, -1},
  {7, 5, 2, 7, 2, 11, 5, 9, 2, 3, 2, 8, 9, 8, 2, -1},
  {2, 5, 10, 2, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1},
  {8, 2, 0, 8, 5, 2, 8, 7, 5, 10, 2, 5, -1, -1, -1, -1},
  {9, 0, 1, 5, 10, 3, 5, 3, 7, 3, 10, 2, -1, -1, -1, -1},
  {9, 8, 2, 9, 2, 1, 8, 7, 2, 10, 2, 5, 7, 5, 2, -1},
  {1, 3, 5, 3, 7, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 7, 0, 7, 1, 1, 7, 5, -1, -1, -1, -1, -1, -1, -1},
  {9, 0, 3, 9, 3, 5, 5, 3, 7, -1, -1, -1, -1, -1, -1, -1},
  {9, 8, 7, 5, 9, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {5, 8, 4, 5, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1},
  {5, 0, 4, 5, 11, 0, 5, 10, 11, 11, 3, 0, -1, -1, -1, -1},
  {0, 1, 9, 8, 4, 10, 8, 10, 11, 10, 4, 5, -1, -1, -1, -1},
  {10, 11, 4, 10, 4, 5, 11, 3, 4, 9, 4, 1, 3, 1, 4, -1},
  {2, 5, 1, 2, 8, 5, 2, 11, 8, 4, 5, 8, -1, -1, -1, -1},
  {0, 4, 11, 0, 11, 3, 4, 5, 11, 2, 11, 1, 5, 1, 11, -1},
  {0, 2, 5, 0, 5, 9, 2, 11, 5, 4, 5, 8, 11, 8, 5, -1},
  {9, 4, 5, 2, 11, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 5, 10, 3, 5, 2, 3, 4, 5, 3, 8, 4, -1, -1, -1, -1},
  {5, 10, 2, 5, 2, 4, 4, 2, 0, -1, -1, -1, -1, -1, -1, -1},
  {3, 10, 2, 3, 5, 10, 3, 8, 5, 4, 5, 8, 0, 1, 9, -1},
  {5, 10, 2, 5, 2, 4, 1, 9, 2, 9, 4, 2, -1, -1, -1, -1},
  {8, 4, 5, 8, 5, 3, 3, 5, 1, -1, -1, -1, -1, -1, -1, -1},
  {0, 4, 5, 1, 0, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {8, 4, 5, 8, 5, 3, 9, 0, 5, 0, 3, 5, -1, -1, -1, -1},
  {9, 4, 5, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 11, 7, 4, 9, 11, 9, 10, 11, -1, -1, -1, -1, -1, -1, -1},
  {0, 8, 3, 4, 9, 7, 9, 11, 7, 9, 10, 11, -1, -1, -1, -1},
  {1, 10, 11, 1, 11, 4, 1, 4, 0, 7, 4, 11, -1, -1, -1, -1},
  {3, 1, 4, 3, 4, 8, 1, 10, 4, 7, 4, 11, 10, 11, 4, -1},
  {4, 11, 7, 9, 11, 4, 9, 2, 11, 9, 1, 2, -1, -1, -1, -1},
  {9, 7, 4, 9, 11, 7, 9, 1, 11, 2, 11, 1, 0, 8, 3, -1},
  {11, 7, 4, 11, 4, 2, 2, 4, 0, -1, -1, -1, -1, -1, -1, -1},
  {11, 7, 4, 11, 4, 2, 8, 3, 4, 3, 2, 4, -1, -1, -1, -1},
  {2, 9, 10, 2, 7, 9, 2, 3, 7, 7, 4, 9, -1, -1, -1, -1},
  {9, 10, 7, 9, 7, 4, 10, 2, 7, 8, 7, 0, 2, 0, 7, -1},
  {3, 7, 10, 3, 10, 2, 7, 4, 10, 1, 10, 0, 4, 0, 10, -1},
  {1, 10, 2, 8, 7, 4, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 9, 1, 4, 1, 7, 7, 1, 3, -1, -1, -1, -1, -1, -1, -1},
  {4, 9, 1, 4, 1, 7, 0, 8, 1, 8, 7, 1, -1, -1, -1, -1},
  {4, 0, 3, 7, 4, 3, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {4, 8, 7, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {9, 10, 8, 10, 11, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 0, 9, 3, 9, 11, 11, 9, 10, -1, -1, -1, -1, -1, -1, -1},
  {0, 1, 10, 0, 10, 8, 8, 10, 11, -1, -1, -1, -1, -1, -1, -1},
  {3, 1, 10, 11, 3, 10, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 2, 11, 1, 11, 9, 9, 11, 8, -1, -1, -1, -1, -1, -1, -1},
  {3, 0, 9, 3, 9, 11, 1, 2, 9, 2, 11, 9, -1, -1, -1, -1},
  {0, 2, 11, 8, 0, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {3, 2, 11, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 3, 8, 2, 8, 10, 10, 8, 9, -1, -1, -1, -1, -1, -1, -1},
  {9, 10, 2, 0, 9, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {2, 3, 8, 2, 8, 10, 0, 1, 8, 1, 10, 8, -1, -1, -1, -1},
  {1, 10, 2, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {1, 3, 8, 9, 1, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 9, 1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {0, 3, 8, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
  {-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1},
};
// clang-format on
#endif

// Return the index of the node where the iso-level cuts the edge between
// neighbouring cells a_iv1 and a_iv2, creating it if this rank has not seen
// the edge yet. a_data holds the output fields, iso field first.
int
getEdgeNode(
  LocalSurface& a_surf,
  int a_lev,
  const amrex::IntVect& a_iv1,
  const amrex::IntVect& a_iv2,
  const amrex::Array4<const amrex::Real>& a_data,
  const amrex::GeometryData& a_geom,
  int a_nFields,
  amrex::Real a_isoVal)
{
  EdgeKey key{};
  key[0] = a_lev;
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    key[1 + d] = std::min(a_iv1[d], a_iv2[d]);
    if (a_iv1[d] != a_iv2[d]) {
      key[keySize - 1] = d;
    }
  }
  const auto found = a_surf.edgeToNode.find(key);
  if (found != a_surf.edgeToNode.end()) {
    return found->second;
  }

  const amrex::Real v1 = a_data(a_iv1, 0);
  const amrex::Real v2 = a_data(a_iv2, 0);
  // Same snapping logic as PeleAnalysis' VI_doIt
  amrex::Real mu = 0.0;
  if (std::abs(a_isoVal - v1) < isoEps) {
    mu = 0.0;
  } else if (std::abs(a_isoVal - v2) < isoEps) {
    mu = 1.0;
  } else if (std::abs(v1 - v2) < isoEps) {
    mu = 0.0;
  } else {
    mu = (a_isoVal - v1) / (v2 - v1);
  }

  const int nNodeComp = AMREX_SPACEDIM + a_nFields;
  const int nodeID = static_cast<int>(a_surf.nodes.size() / nNodeComp);
  const amrex::Real* plo = a_geom.ProbLo();
  const amrex::Real* dx = a_geom.CellSize();
  for (int d = 0; d < AMREX_SPACEDIM; ++d) {
    const amrex::Real x1 = plo[d] + (a_iv1[d] + 0.5) * dx[d];
    const amrex::Real x2 = plo[d] + (a_iv2[d] + 0.5) * dx[d];
    a_surf.nodes.push_back(x1 + mu * (x2 - x1));
  }
  for (int n = 0; n < a_nFields; ++n) {
    const amrex::Real p1 = a_data(a_iv1, n);
    const amrex::Real p2 = a_data(a_iv2, n);
    a_surf.nodes.push_back(p1 + mu * (p2 - p1));
  }
  a_surf.keys.insert(a_surf.keys.end(), key.begin(), key.end());
  a_surf.edgeToNode.emplace(key, nodeID);
  return nodeID;
}

// Append an element, skipping degenerate ones, and accumulate its measure
// (length in 2D, area in 3D).
void
addElement(
  LocalSurface& a_surf,
  const std::array<int, nodesPerElt>& a_ids,
  int a_nNodeComp)
{
  for (int a = 0; a < nodesPerElt; ++a) {
    for (int b = a + 1; b < nodesPerElt; ++b) {
      if (a_ids[a] == a_ids[b]) {
        return;
      }
    }
  }
  a_surf.elts.insert(a_surf.elts.end(), a_ids.begin(), a_ids.end());

  const amrex::Real* p0 = &a_surf.nodes[a_ids[0] * a_nNodeComp];
  const amrex::Real* p1 = &a_surf.nodes[a_ids[1] * a_nNodeComp];
#if AMREX_SPACEDIM == 2
  a_surf.measure += std::sqrt(
    (p1[0] - p0[0]) * (p1[0] - p0[0]) + (p1[1] - p0[1]) * (p1[1] - p0[1]));
#else
  const amrex::Real* p2 = &a_surf.nodes[a_ids[2] * a_nNodeComp];
  const amrex::Real ax = p1[0] - p0[0];
  const amrex::Real ay = p1[1] - p0[1];
  const amrex::Real az = p1[2] - p0[2];
  const amrex::Real bx = p2[0] - p0[0];
  const amrex::Real by = p2[1] - p0[1];
  const amrex::Real bz = p2[2] - p0[2];
  const amrex::Real cx = ay * bz - az * by;
  const amrex::Real cy = az * bx - ax * bz;
  const amrex::Real cz = ax * by - ay * bx;
  a_surf.measure += 0.5 * std::sqrt(cx * cx + cy * cy + cz * cz);
#endif
}

#ifdef AMREX_USE_GPU
// Copy of a (i)MultiFab, ghost cells included, in host-accessible memory
template <class MF>
MF
pinnedCopy(const MF& a_src)
{
  MF dst(
    a_src.boxArray(), a_src.DistributionMap(), a_src.nComp(), a_src.nGrowVect(),
    amrex::MFInfo().SetArena(amrex::The_Pinned_Arena()));
  amrex::Copy(dst, a_src, 0, 0, a_src.nComp(), a_src.nGrowVect());
  return dst;
}
#endif

// Gather a_loc from every rank onto the IO rank, concatenated in rank order.
// On the IO rank, a_disp[r] is where rank r's data starts (a_disp has NProcs+1
// entries, the last one being the total size).
template <class T>
std::vector<T>
gatherToIO(const std::vector<T>& a_loc, std::vector<int>& a_disp)
{
  namespace PD = amrex::ParallelDescriptor;
  const int ioProc = PD::IOProcessorNumber();
  const int nLoc = static_cast<int>(a_loc.size());
  const std::vector<int> counts = PD::Gather(nLoc, ioProc);
  std::vector<T> all;
  a_disp.clear();
  if (PD::IOProcessor()) {
    a_disp.resize(counts.size() + 1, 0);
    for (std::size_t r = 0; r < counts.size(); ++r) {
      a_disp[r + 1] = a_disp[r] + counts[r];
    }
    all.resize(a_disp.back());
  }
  PD::Gatherv(a_loc.data(), nLoc, all.data(), counts, a_disp, ioProc);
  return all;
}

} // namespace

void
DiagIsosurface::init(const std::string& a_prefix, std::string_view a_diagName)
{
  DiagBase::init(a_prefix, a_diagName);

  amrex::ParmParse pp(a_prefix);
  pp.get("field_name", m_isoFieldName);
  pp.get("value", m_isoValue);

  m_fieldNames.clear();
  m_fieldNames.push_back(m_isoFieldName);
  const int nFields = pp.countval("field_names");
  for (int n = 0; n < nFields; ++n) {
    std::string name;
    pp.get("field_names", name, n);
    if (
      std::find(m_fieldNames.begin(), m_fieldNames.end(), name) ==
      m_fieldNames.end()) {
      m_fieldNames.push_back(name);
    }
  }

  pp.query_enum_case_insensitive("format", m_format);
  pp.query("write_measure", m_writeMeasure);
  pp.query("verbose", m_verbose);
}

void
DiagIsosurface::addVars(amrex::Vector<std::string>& a_varList)
{
  DiagBase::addVars(a_varList);
  for (const auto& name : m_fieldNames) {
    a_varList.push_back(name);
  }
}

void
DiagIsosurface::prepare(
  int a_nlevels,
  const amrex::Vector<amrex::Geometry>& a_geoms,
  const amrex::Vector<amrex::BoxArray>& a_grids,
  const amrex::Vector<amrex::DistributionMapping>& a_dmap,
  const amrex::Vector<std::string>& a_varNames)
{
  if (first_time) {
    DiagBase::prepare(a_nlevels, a_geoms, a_grids, a_dmap, a_varNames);
    first_time = false;
  }

  m_geoms.resize(a_nlevels);
  m_refRatio.resize(a_nlevels - 1);
  for (int lev = 0; lev < a_nlevels; lev++) {
    m_geoms[lev] = a_geoms[lev];
    if (lev > 0) {
      amrex::IntVect rr;
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        rr[d] = static_cast<int>(
          std::lround(a_geoms[lev - 1].CellSize(d) / a_geoms[lev].CellSize(d)));
      }
      m_refRatio[lev - 1] = rr;
    }
  }
}

void
DiagIsosurface::processDiag(
  int a_nstep,
  const amrex::Real& a_time,
  const amrex::Vector<const amrex::MultiFab*>& a_state,
  const amrex::Vector<std::string>& a_stateVar)
{
  BL_PROFILE("DiagIsosurface::processDiag()");

  const int nlev = static_cast<int>(a_state.size());
  const int nOut = static_cast<int>(m_fieldNames.size());
  const int nFilt = static_cast<int>(m_filters.size());
  const int nNC = nNodeComp();

  amrex::Vector<int> stateComp(nOut);
  for (int n = 0; n < nOut; ++n) {
    stateComp[n] = getFieldIndex(m_fieldNames[n], a_stateVar);
  }

  LocalSurface surf;

  for (int lev = 0; lev < nlev; ++lev) {
    const amrex::Geometry& geom = m_geoms[lev];
    const amrex::MultiFab& state = *a_state[lev];
    const amrex::BoxArray& ba = state.boxArray();
    const amrex::DistributionMapping& dm = state.DistributionMap();
    const amrex::Periodicity period = geom.periodicity();

    // ------------------------------------------------------------------
    // Working copy with one ghost cell: output fields (iso field first),
    // then filter fields. Ghost cells come from the caller where available,
    // and FillBoundary takes care of same-level and periodic neighbours.
    // ------------------------------------------------------------------
    const int ngState = state.nGrowVect().min();
    const int ngCopy = std::min(ngState, 1);
    amrex::MultiFab work(ba, dm, nOut + nFilt, 1);
    if (ngCopy == 0) {
      work.setBndry(0.0);
    }
    for (int n = 0; n < nOut; ++n) {
      amrex::MultiFab::Copy(work, state, stateComp[n], n, 1, ngCopy);
    }
    for (int f = 0; f < nFilt; ++f) {
      amrex::MultiFab::Copy(
        work, state, m_filters[f].m_fdata.m_filterVarIdx, nOut + f, 1, ngCopy);
    }
    work.FillBoundary(period);

    // ------------------------------------------------------------------
    // Corner flags. levelBit: the cell belongs to the level (valid cell, or
    // ghost cell overlapping another box, possibly through periodicity).
    // usableBit: the cell is not fine-covered, not outside a non-periodic
    // boundary, holds data (without ghost cells from the caller only
    // same-level ghosts do), and passes the filters.
    // ------------------------------------------------------------------
    amrex::iMultiFab cellType(ba, dm, 1, 1);
    cellType.BuildMask(geom.Domain(), period, 1, 0, 2, 1);

    amrex::iMultiFab flags;
    if (lev < nlev - 1) {
      flags = amrex::makeFineMask(
        ba, dm, amrex::IntVect(1), a_state[lev + 1]->boxArray(),
        m_refRatio[lev], period, 1, 0);
    } else {
      flags.define(ba, dm, 1, 1);
      flags.setVal(1);
    }

    const bool ghostsFilled = ngState >= 1;
    auto const& flagArr = flags.arrays();
    auto const& typeArr = cellType.const_arrays();
    auto const& workArr = work.const_arrays();
    const auto* fdata_p = m_filterData.data();
    amrex::ParallelFor(
      flags, amrex::IntVect(1),
      [=] AMREX_GPU_DEVICE(int box_no, int i, int j, int k) noexcept {
        const int type = typeArr[box_no](i, j, k);
        bool usable = flagArr[box_no](i, j, k) != 0 && type != 2 &&
                      (ghostsFilled || type == 1);
        for (int f = 0; f < nFilt; ++f) {
          const amrex::Real v = workArr[box_no](i, j, k, nOut + f);
          if (v < fdata_p[f].m_low_val || v > fdata_p[f].m_high_val) {
            usable = false;
          }
        }
        flagArr[box_no](i, j, k) =
          (usable ? usableBit : 0) | (type == 1 ? levelBit : 0);
      });

    // The marching itself runs on the host.
#ifdef AMREX_USE_GPU
    const amrex::MultiFab hwork = pinnedCopy(work);
    const amrex::iMultiFab hflags = pinnedCopy(flags);
#else
    const amrex::MultiFab& hwork = work;
    const amrex::iMultiFab& hflags = flags;
#endif
    amrex::Gpu::streamSynchronize();

    // ------------------------------------------------------------------
    // March over the cubes owned by each box: a cube is owned by the box
    // holding, as a valid cell, the first of its corners (in corner order)
    // that belongs to the level. Periodic images share that corner, so every
    // cube of the level is processed exactly once. Looping over the box
    // grown by one on the low side visits every cube touching the box,
    // including those based in a coarse-fine ghost cell.
    // ------------------------------------------------------------------
    const auto geomdata = geom.data();
    for (amrex::MFIter mfi(hwork); mfi.isValid(); ++mfi) {
      const amrex::Box& vbx = mfi.validbox();
      auto const& data = hwork.const_array(mfi);
      auto const& flg = hflags.const_array(mfi);

      amrex::Box loopBox = vbx;
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        loopBox.growLo(d, 1);
      }

      amrex::LoopOnCpu(loopBox, [&](int i, int j, int k) {
        const amrex::IntVect base(AMREX_D_DECL(i, j, k));
        amrex::ignore_unused(k);

        std::array<amrex::IntVect, nCorners> corner;
        int caseIdx = 0;
        for (int c = 0; c < nCorners; ++c) {
          corner[c] = base + amrex::IntVect(AMREX_D_DECL(
                               cornerOffset[c][0], cornerOffset[c][1],
                               cornerOffset[c][2]));
          if (data(corner[c], 0) < m_isoValue) {
            caseIdx |= (1 << c);
          }
        }
        if (caseIdx == 0 || caseIdx == (1 << nCorners) - 1) {
          return;
        }

        bool ownerFound = false;
        for (int c = 0; c < nCorners; ++c) {
          const int f = flg(corner[c]);
          if ((f & usableBit) == 0) {
            return;
          }
          if (!ownerFound && (f & levelBit) != 0) {
            if (!vbx.contains(corner[c])) {
              return;
            }
            ownerFound = true;
          }
        }

        auto node = [&](int e) {
          return getEdgeNode(
            surf, lev, corner[edgeCorners[e][0]], corner[edgeCorners[e][1]],
            data, geomdata, nOut, m_isoValue);
        };

#if AMREX_SPACEDIM == 2
        for (int s = 0; s < 4 && segTable[caseIdx][s] != -1; s += 2) {
          std::array<int, nodesPerElt> ids{
            node(segTable[caseIdx][s]), node(segTable[caseIdx][s + 1])};
          addElement(surf, ids, nNC);
        }
#else
        std::array<int, nEdges> vert{};
        for (int e = 0; e < nEdges; ++e) {
          if ((edgeTable[caseIdx] & (1 << e)) != 0) {
            vert[e] = node(e);
          }
        }
        for (int t = 0; t < 16 && triTable[caseIdx][t] != -1; t += 3) {
          std::array<int, nodesPerElt> ids{
            vert[triTable[caseIdx][t]], vert[triTable[caseIdx][t + 1]],
            vert[triTable[caseIdx][t + 2]]};
          addElement(surf, ids, nNC);
        }
#endif
      });
    }
  }
  surf.edgeToNode.clear();

  // Measure: each cube is owned by exactly one box, so the local sums add up
  // without double counting.
  amrex::Real measure = surf.measure;
  amrex::Long nElts = static_cast<amrex::Long>(surf.elts.size()) / nodesPerElt;
  const int ioProc = amrex::ParallelDescriptor::IOProcessorNumber();
  amrex::ParallelDescriptor::ReduceRealSum(measure, ioProc);
  amrex::ParallelDescriptor::ReduceLongSum(nElts, ioProc);

  if (m_verbose > 0) {
    amrex::Print() << "  DiagIsosurface " << m_diagfile << ": "
                   << m_isoFieldName << " = " << m_isoValue << ", " << nElts
                   << " elements, "
                   << (AMREX_SPACEDIM == 2 ? "length = " : "area = ") << measure
                   << "\n";
  }

  if (m_writeMeasure) {
    writeMeasure(a_nstep, a_time, measure, nElts);
  }

  if (m_format != IsoSurfFormat::None) {
    collateAndWrite(surf.nodes, surf.keys, surf.elts, a_nstep, a_time);
  }
}

void
DiagIsosurface::collateAndWrite(
  const std::vector<amrex::Real>& a_nodes,
  const std::vector<int>& a_keys,
  const std::vector<int>& a_elts,
  int a_nstep,
  const amrex::Real& a_time) const
{
  BL_PROFILE("DiagIsosurface::collateAndWrite()");

  namespace PD = amrex::ParallelDescriptor;

  // MPI_Gatherv takes int counts and displacements: check that the gathered
  // arrays fit before narrowing anything to int.
  amrex::Long totSize = static_cast<amrex::Long>(
    std::max({a_nodes.size(), a_keys.size(), a_elts.size()}));
  PD::ReduceLongSum(totSize);
  if (totSize > static_cast<amrex::Long>(std::numeric_limits<int>::max())) {
    amrex::Abort(
      "DiagIsosurface " + m_diagfile +
      ": surface too large to gather on one rank for output; use format = "
      "None to only compute the measure");
  }

  std::vector<int> dispNode, dispKey, dispElt;
  const std::vector<amrex::Real> nodesAll = gatherToIO(a_nodes, dispNode);
  const std::vector<int> keysAll = gatherToIO(a_keys, dispKey);
  std::vector<int> eltsAll = gatherToIO(a_elts, dispElt);

  if (!PD::IOProcessor()) {
    return;
  }

  // Merge the nodes shared between ranks (same level and same edge) and
  // renumber the elements accordingly.
  const int nNC = nNodeComp();
  const int nNodesTot = static_cast<int>(keysAll.size()) / keySize;
  EdgeMap keyToNode;
  keyToNode.reserve(nNodesTot);
  std::vector<int> remap(nNodesTot);
  std::vector<amrex::Real> nodes;
  nodes.reserve(nodesAll.size());
  for (int g = 0; g < nNodesTot; ++g) {
    EdgeKey key;
    std::copy_n(&keysAll[g * keySize], keySize, key.begin());
    const int next = static_cast<int>(keyToNode.size());
    const auto res = keyToNode.emplace(key, next);
    if (res.second) {
      nodes.insert(
        nodes.end(), nodesAll.begin() + g * nNC,
        nodesAll.begin() + (g + 1) * nNC);
    }
    remap[g] = res.first->second;
  }

  for (int r = 0; r < PD::NProcs(); ++r) {
    const int nodeOffset = dispKey[r] / keySize;
    for (int n = dispElt[r]; n < dispElt[r + 1]; ++n) {
      eltsAll[n] = remap[nodeOffset + eltsAll[n]];
    }
  }

  if (eltsAll.empty()) {
    amrex::Print() << "  DiagIsosurface " << m_diagfile
                   << ": empty iso-surface, nothing written\n";
    return;
  }

  const std::string base = outputBase(a_nstep, a_time);
  if (m_format == IsoSurfFormat::XDMF) {
    writeXDMF(base, a_time, nodes, eltsAll);
  } else if (m_format == IsoSurfFormat::MEF) {
    writeMEF(base, a_time, nodes, eltsAll);
  }
}

std::string
DiagIsosurface::outputBase(int a_nstep, const amrex::Real& a_time) const
{
  std::string base = m_diagfile;
  if (m_interval > 0) {
    base = amrex::Concatenate(m_diagfile, a_nstep, 6);
  }
  if (m_per > 0.0) {
    base = m_diagfile + std::to_string(a_time);
  }
  return base;
}

void
DiagIsosurface::writeXDMF(
  const std::string& a_base,
  const amrex::Real& a_time,
  const std::vector<amrex::Real>& a_nodes,
  const std::vector<int>& a_elts) const
{
  const int nNC = nNodeComp();
  const std::size_t nNodes = a_nodes.size() / nNC;
  const std::size_t nElts = a_elts.size() / nodesPerElt;
  const std::string binFile = a_base + ".bin";
  // The .xmf references the binary file relative to its own location
  const std::string binName = binFile.substr(binFile.find_last_of('/') + 1);

  // Binary layout: connectivity, interleaved coordinates, then one block per
  // field.
  {
    std::ofstream ofs(
      binFile, std::ios::out | std::ios::trunc | std::ios::binary);
    if (!ofs.good()) {
      amrex::FileOpenFailed(binFile);
    }
    ofs.write(
      reinterpret_cast<const char*>(a_elts.data()),
      static_cast<std::streamsize>(sizeof(int) * a_elts.size()));
    std::vector<amrex::Real> buf(nNodes * AMREX_SPACEDIM);
    for (std::size_t i = 0; i < nNodes; ++i) {
      for (int d = 0; d < AMREX_SPACEDIM; ++d) {
        buf[i * AMREX_SPACEDIM + d] = a_nodes[i * nNC + d];
      }
    }
    ofs.write(
      reinterpret_cast<const char*>(buf.data()),
      static_cast<std::streamsize>(sizeof(amrex::Real) * buf.size()));
    buf.resize(nNodes);
    for (int n = AMREX_SPACEDIM; n < nNC; ++n) {
      for (std::size_t i = 0; i < nNodes; ++i) {
        buf[i] = a_nodes[i * nNC + n];
      }
      ofs.write(
        reinterpret_cast<const char*>(buf.data()),
        static_cast<std::streamsize>(sizeof(amrex::Real) * nNodes));
    }
  }

  std::ofstream xmf(a_base + ".xmf");
  if (!xmf.good()) {
    amrex::FileOpenFailed(a_base + ".xmf");
  }
  std::size_t seek = sizeof(int) * a_elts.size();
  // Binary Real data item starting at byte 'seek' of the .bin file
  auto floatItem = [&](std::size_t a_dims) {
    xmf << "        <DataItem Format=\"Binary\" DataType=\"Float\" "
           "Precision=\""
        << sizeof(amrex::Real) << "\" Endian=\"Native\" Seek=\"" << seek
        << "\" Dimensions=\"" << a_dims << "\">" << binName << "</DataItem>\n";
    seek += sizeof(amrex::Real) * a_dims;
  };

  xmf << std::setprecision(17);
  xmf << "<?xml version=\"1.0\"?>\n";
  xmf << "<Xdmf Version=\"3.0\" "
         "xmlns:xi=\"http://www.w3.org/2001/XInclude\">\n";
  xmf << "  <Domain>\n";
  xmf << "    <Grid Name=\"isoSurface\">\n";
  xmf << "      <Information Name=\"Variable\" Value=\"" << m_isoFieldName
      << "\"/>\n";
  xmf << "      <Information Name=\"IsoValue\" Value=\"" << m_isoValue
      << "\"/>\n";
  xmf << "      <Time Value=\"" << a_time << "\"/>\n";
#if AMREX_SPACEDIM == 2
  xmf << "      <Topology TopologyType=\"Polyline\" NodesPerElement=\"2\" "
         "NumberOfElements=\""
      << nElts << "\">\n";
#else
  xmf << "      <Topology TopologyType=\"Triangle\" NumberOfElements=\""
      << nElts << "\">\n";
#endif
  xmf << "        <DataItem Format=\"Binary\" DataType=\"Int\" "
         "Precision=\"4\" Endian=\"Native\" Dimensions=\""
      << a_elts.size() << "\">" << binName << "</DataItem>\n";
  xmf << "      </Topology>\n";
  xmf << "      <Geometry GeometryType=\""
      << (AMREX_SPACEDIM == 2 ? "XY" : "XYZ") << "\">\n";
  floatItem(nNodes * AMREX_SPACEDIM);
  xmf << "      </Geometry>\n";
  for (const auto& name : m_fieldNames) {
    xmf << "      <Attribute Name=\"" << name
        << "\" AttributeType=\"Scalar\" Center=\"Node\">\n";
    floatItem(nNodes);
    xmf << "      </Attribute>\n";
  }
  xmf << "    </Grid>\n";
  xmf << "  </Domain>\n";
  xmf << "</Xdmf>\n";
}

void
DiagIsosurface::writeMEF(
  const std::string& a_base,
  const amrex::Real& a_time,
  const std::vector<amrex::Real>& a_nodes,
  const std::vector<int>& a_elts) const
{
  // Same layout as PeleAnalysis' isosurface tool, so its MEF utilities can
  // read the output: label, variable names, element count, a FAB holding the
  // nodes (node-major), then 1-based connectivity.
  const int nNC = nNodeComp();
  const int nNodes = static_cast<int>(a_nodes.size()) / nNC;
  const std::size_t nElts = a_elts.size() / nodesPerElt;

  std::string vars = AMREX_SPACEDIM == 2 ? "X Y" : "X Y Z";
  for (const auto& name : m_fieldNames) {
    vars += " " + name;
  }

  amrex::FArrayBox fab(
    amrex::Box(
      amrex::IntVect::TheZeroVector(),
      amrex::IntVect(AMREX_D_DECL(nNodes - 1, 0, 0))),
    nNC, amrex::The_Cpu_Arena());
  std::copy(a_nodes.begin(), a_nodes.end(), fab.dataPtr());

  std::vector<int> elts(a_elts.size());
  for (std::size_t n = 0; n < a_elts.size(); ++n) {
    elts[n] = a_elts[n] + 1;
  }

  const std::string fileName = a_base + ".mef";
  std::ofstream ofs(
    fileName, std::ios::out | std::ios::trunc | std::ios::binary);
  if (!ofs.good()) {
    amrex::FileOpenFailed(fileName);
  }
  ofs << std::setprecision(17) << a_time << '\n';
  ofs << vars << '\n';
  ofs << nElts << " " << nodesPerElt << '\n';
  fab.writeOn(ofs);
  ofs.write(
    reinterpret_cast<const char*>(elts.data()),
    static_cast<std::streamsize>(sizeof(int) * elts.size()));
}

void
DiagIsosurface::writeMeasure(
  int a_nstep,
  const amrex::Real& a_time,
  amrex::Real a_measure,
  amrex::Long a_nElts)
{
  if (!amrex::ParallelDescriptor::IOProcessor()) {
    return;
  }
  const std::string fileName = m_diagfile + "_measure.dat";
  bool writeHeader = false;
  if (!m_measureHeaderDone) {
    // Don't repeat the header when appending to an existing file (restart)
    std::ifstream probe(fileName);
    writeHeader =
      !probe.good() || probe.peek() == std::ifstream::traits_type::eof();
    m_measureHeaderDone = true;
  }
  std::ofstream ofs(fileName, std::ios::out | std::ios::app);
  if (!ofs.good()) {
    amrex::FileOpenFailed(fileName);
  }
  if (writeHeader) {
    ofs << "# " << m_isoFieldName << " = " << m_isoValue << "\n";
    ofs << "# step time " << (AMREX_SPACEDIM == 2 ? "length" : "area")
        << " n_elements\n";
  }
  ofs << a_nstep << " " << std::setprecision(17) << a_time << " " << a_measure
      << " " << a_nElts << "\n";
}

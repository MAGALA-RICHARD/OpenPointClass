//
// Created by rmagala on 9/25/2026.
//

#ifndef OPC_HELPER_HPP
#define OPC_HELPER_HPP

#include <string>
#include <vector>
#include <map>

namespace helpers
    // splits a given point cloud based on the length of xy origin
    {std::vector<std::string> tile_cloud(const std::string& input,
                    const std::string& output,
                    double length = 50.0,
                    double buffer = 0.0,
                    double origin_x = 0.0,
                    double origin_y = 0.0);

     // merges multiple point clouds
    std::string merge_clouds(const std::vector<std::string>& inputs,
                             const std::string& output);
    struct FootprintResult
    {
        double areaSquareUnits;
        std::string boundaryWkt;
    };
 // calculate the area
    FootprintResult computeFootprint(const std::string& input,
                                     double edgeLength,
                                     int minPointsPerHexagon = 1);

  //  clips the Quadrats
  std::vector<std::string> clipQuadrats(
          const std::string& input_file,
          double center_x,
          double center_y,
          const std::vector<std::string>& quadrats,
          const std::map<std::string, int>& quadrat_ids,
          double length,
          const std::string& output_dir);

}

#endif // OPC_HELPER_HPP
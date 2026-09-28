//
// Created by rmagala on 9/25/2026.
//

#ifndef OPC_QUADRAT_HPP
#define OPC_QUADRAT_HPP

#endif //OPC_QUADRAT_HPP
#pragma once

#include <map>
#include <string>
#include <vector>


struct Bounds
{
    double xmin;
    double ymin;
    double xmax;
    double ymax;
};

std::map<std::string, Bounds> getQuadratsBounds(
        double center_x,
        double center_y,
        const std::vector<std::string>& quadrats,
        double length
);

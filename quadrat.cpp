//
// Created by rmagala on 9/25/2026.
//

#include "quadrat.hpp"

#include <algorithm>
#include <cctype>
#include <iostream>
#include <map>
#include <stdexcept>
#include <string>
#include <vector>

std::map<std::string, Bounds> getQuadratsBounds(
        double center_x,
        double center_y,
        const std::vector<std::string>& quadrats,
        double length
)
{
    std::map<std::string, Bounds> directions = {
            {"NW", {center_x - length, center_y,          center_x,          center_y + length}},
            {"NE", {center_x,          center_y,          center_x + length, center_y + length}},
            {"SW", {center_x - length, center_y - length, center_x,          center_y}},
            {"SE", {center_x,          center_y - length, center_x + length, center_y}}
    };

    std::map<std::string, Bounds> result;

    for (auto q : quadrats)
    {
        std::transform(
                q.begin(),
                q.end(),
                q.begin(),
                [](unsigned char c)
                {
                    return static_cast<char>(std::toupper(c));
                }
        );

        const auto it = directions.find(q);

        if (it == directions.end())
        {
            throw std::invalid_argument(
                    "Unknown quadrat '" + q +
                    "'. Use NW, NE, SW, or SE."
            );
        }

        result[q] = it->second;
    }

    return result;
}


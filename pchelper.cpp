//
// Created by rmagala on 9/25/2026.
//

#include "pchelper.hpp"
#include "helper.hpp"

#include <exception>
#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <algorithm>
#include <cmath>

int main(int argc, char* argv[])
{
    const auto usage = [] {
        std::cout
                << "pchelper tile <input> <output_#.laz> <length> [buffer]\n"
                << "pchelper merge <output> <input1> [input2 ...]\n"
                << "pchelper footprint <input> <edge_length> [threshold]\n"
                << "pchelper quadrant <output_dir> <input.las> <center_x> <center_y> <length> <DIR:ID> [DIR:ID ...]\n";
    };
    if (argc < 2 || std::string(argv[1]) == "--help")
    {
        usage();
        return argc < 2 ? 1 : 0;
    }

    try
    {
        const std::string command = argv[1];
        if (command == "tile" && (argc == 5 || argc == 6))
        {
            const auto files = helpers::tile_cloud(
                    argv[2], argv[3], std::stod(argv[4]),
                    argc == 6 ? std::stod(argv[5]) : 0.0, 0.0, 0.0);
            for (const auto& file : files)
                std::cout << file << '\n';
        }
        else if (command == "merge" && argc >= 4)
        {
            const std::vector<std::string> inputs(argv + 3, argv + argc);
            std::cout << helpers::merge_clouds(inputs, argv[2]) << '\n';
        }
        else if (command == "quadrant") {
            if (argc < 8) {
                std::cerr << "Usage: pchelper quadrant <output_dir> <input.las> "
                             "<center_x> <center_y> <length> <DIR:ID> [DIR:ID ...]\n";
                return 1;
            }

            try {
                const auto parse_number = [](const std::string& text) {
                    std::size_t used = 0;
                    const double value = std::stod(text, &used);
                    if (used != text.size() || !std::isfinite(value))
                        throw std::invalid_argument("Invalid number: " + text);
                    return value;
                };

                const std::string output_dir = argv[2], input_file = argv[3];
                const double center_x = parse_number(argv[4]), center_y = parse_number(argv[5]);
                const double length = parse_number(argv[6]);

                if (length <= 0.0)
                    throw std::invalid_argument("Quadrat length must be greater than zero.");

                std::vector<std::string> quadrats;
                std::map<std::string, int> quadrat_ids;

                for (int i = 7; i < argc; ++i) {
                    const std::string argument = argv[i];
                    const auto colon = argument.find(':');

                    if (colon == std::string::npos || colon == 0 || colon + 1 == argument.size())
                        throw std::invalid_argument("Expected DIR:ID, such as NW:001. Received: " + argument);

                    std::string direction = argument.substr(0, colon);
                    std::transform(direction.begin(), direction.end(), direction.begin(),
                                   [](unsigned char c) { return static_cast<char>(std::toupper(c)); });

                    if (direction != "NW" && direction != "NE" && direction != "SW" && direction != "SE")
                        throw std::invalid_argument("Unknown quadrat direction: " + direction);

                    const std::string id_text = argument.substr(colon + 1);
                    std::size_t used = 0;
                    const int id = std::stoi(id_text, &used, 10);

                    if (used != id_text.size() || id < 0)
                        throw std::invalid_argument("QuadratID must be a nonnegative integer: " + id_text);
                    if (!quadrat_ids.emplace(direction, id).second)
                        throw std::invalid_argument("Repeated quadrat direction: " + direction);

                    quadrats.push_back(direction);
                }

                const auto files = helpers::clipQuadrats(input_file, center_x, center_y,
                                                         quadrats, quadrat_ids, length, output_dir);
                for (const auto& file : files) std::cout << file << '\n';
            } catch (const std::exception& error) {
                std::cerr << "pchelper: " << error.what() << '\n';
                return 1;
            }
        }
        else if (command == "footprint" && (argc == 4 || argc == 5))
        {
            const auto result = helpers::computeFootprint(
                    argv[2], std::stod(argv[3]),
                    argc == 5 ? std::stoi(argv[4]) : 1);
            std::cout << std::setprecision(17)
                      << "Area (square coordinate units): "
                      << result.areaSquareUnits << '\n'
                      << "Boundary WKT: " << result.boundaryWkt << '\n';
        }
        else
        {
            usage();
            return 1;
        }
        return 0;
    }
    catch (const std::exception& error)
    {
        std::cerr << "pchelper: " << error.what() << '\n';
        return 1;
    }
}

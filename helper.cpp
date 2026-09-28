#ifdef WITH_PDAL
#include <pdal/Kernel.hpp>
#include <pdal/Log.hpp>
#include <pdal/PluginManager.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/util/FileUtils.hpp>
#include <pdal/Metadata.hpp>
#include <pdal/Options.hpp>
#include <pdal/PipelineManager.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/SpatialReference.hpp>
#include <pdal/Stage.hpp>
#endif
#include "helper.hpp"
#include "quadrat.hpp"
#include <memory>
#include <stdexcept>
#include <stdexcept>
#include <vector>
#include <filesystem>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <memory>
#include <stdexcept>
#include <iostream>
#include <sstream>
#include <map>


namespace helpers
{
    std::vector<std::string> tile_cloud(
            const std::string& input, const std::string& output,
            double length, double buffer, double origin_x, double origin_y)
    {
        pdal::StageFactory factory;
        std::unique_ptr<pdal::Kernel> kernel(
                pdal::PluginManager<pdal::Kernel>::createObject("kernels.tile"));

        if (!kernel)
            throw std::runtime_error("PDAL tile kernel is unavailable.");

        pdal::StringList args{
                "--input=" + input,
                "--output=" + output,
                "--length=" + std::to_string(length),
                "--buffer=" + std::to_string(buffer),
                "--origin_x=" + std::to_string(origin_x),
                "--origin_y=" + std::to_string(origin_y)
        };

        auto log = pdal::Log::makeLog("OpenPointClass tile", "stderr");
        if (kernel->run(args, log) != 0)
            throw std::runtime_error("PDAL tile failed; see its diagnostics.");

        std::string pattern = output;
        std::replace(pattern.begin(), pattern.end(), '#', '*');

        auto files = pdal::FileUtils::glob(pattern);
        std::sort(files.begin(), files.end());
        if (files.empty())
            throw std::runtime_error(
                    "No files were collected after PDAL split input cloud: " + input);
        std::cout << files.size() << " files were created from " << input<< '\n';
        return files;
    }

    std::string merge_clouds(const std::vector<std::string>& inputs,
                             const std::string& output)
    {
        if (inputs.empty())
            throw std::invalid_argument("No input files provided.");

        pdal::PipelineManager manager;
        auto& writer = manager.makeWriter(output, "");

        for (const auto& input : inputs)
            writer.setInput(manager.makeReader(input, ""));

        if (!writer.pipelineStreamable())
            throw std::runtime_error(
                    "Selected input/output formats do not support streaming.");

        pdal::FixedPointTable table(10000);
        writer.prepare(table);
        writer.execute(table);
        if (!std::filesystem::is_regular_file(output))
            throw std::runtime_error("Perhaps! Output file was not created: " + output);

        return output;
    }

    FootprintResult computeFootprint(const std::string& input,
                                     double edgeLength,
                                     int minPointsPerHexagon)
    {
        if (!std::isfinite(edgeLength) || edgeLength <= 0 ||
            minPointsPerHexagon < 1)
            throw std::invalid_argument(
                    "Edge length must be finite and positive; threshold must be >= 1.");

        pdal::PipelineManager manager;
        auto& reader = manager.makeReader(input, "");

        pdal::Options options;
        options.add("edge_length", edgeLength);
        options.add("threshold", minPointsPerHexagon);
        options.add("smooth", false);
        options.add("precision", 17);

        auto& hexbin = manager.makeFilter("filters.hexbin", options);
        hexbin.setInput(reader);

        if (!hexbin.pipelineStreamable())
            throw std::runtime_error("Input does not support streaming.");

        pdal::FixedPointTable table(10000);
        hexbin.prepare(table);

        if (!reader.getSpatialReference().isProjected())
            throw std::invalid_argument(
                    "Footprint requires a known projected CRS: " + input);

        hexbin.execute(table);

        const auto metadata = hexbin.getMetadata();
        const auto error = metadata.findChild("error");
        if (error.valid())
            throw std::runtime_error(error.value<std::string>());

        const auto area = metadata.findChild("area");
        const auto boundary = metadata.findChild("boundary");
        if (!area.valid() || !boundary.valid())
            throw std::runtime_error("No footprint returned for: " + input);

        FootprintResult result{
                area.value<double>(), boundary.value<std::string>()
        };

        if (!std::isfinite(result.areaSquareUnits) ||
            result.areaSquareUnits <= 0 || result.boundaryWkt.empty() ||
            result.boundaryWkt.find("EMPTY") != std::string::npos)
            throw std::runtime_error("No usable footprint for: " + input);

        return result;
    }


        int quadratPositionCode(const std::string& direction)
        {
            if (direction == "NW") return 1;
            if (direction == "NE") return 2;
            if (direction == "SW") return 3;
            if (direction == "SE") return 4;

            throw std::invalid_argument(
                    "Unknown quadrat direction: " + direction
            );
        }

        std::string formatQuadratId(int id)
        {
            std::ostringstream ss;
            ss << std::setw(3) << std::setfill('0') << id;
            return ss.str();
        }


    std::vector<std::string> clipQuadrats(
            const std::string& input_file,
            double center_x,
            double center_y,
            const std::vector<std::string>& quadrats,
            const std::map<std::string, int>& quadrat_ids,
            double length,
            const std::string& output_dir
    )
    {
        if (length <= 0.0)
            throw std::invalid_argument("Quadrat length must be greater than zero.");

        if (!std::filesystem::exists(input_file))
            throw std::runtime_error("Input point cloud does not exist: " + input_file);

        std::filesystem::create_directories(output_dir);

        const auto bounds =
                getQuadratsBounds(center_x, center_y, quadrats, length);

        std::vector<std::string> output_files;

        for (const auto& [direction, b] : bounds)
        {
            const auto id_it = quadrat_ids.find(direction);

            if (id_it == quadrat_ids.end())
                throw std::invalid_argument(
                        "Missing QuadratID for direction '" + direction + "'."
                );

            const int quadrat_id = id_it->second;
            const int position_code = quadratPositionCode(direction);

            const auto output_path =
                    std::filesystem::path(output_dir) /
                    (formatQuadratId(quadrat_id) + "_" + direction + ".las");

            pdal::StageFactory factory;

            auto* reader = factory.createStage("readers.las");
            auto* crop = factory.createStage("filters.crop");
            auto* assign_id = factory.createStage("filters.assign");
            auto* assign_pos = factory.createStage("filters.assign");
            auto* writer = factory.createStage("writers.las");

            if (!reader || !crop || !assign_id || !assign_pos || !writer)
                throw std::runtime_error("Could not create one or more PDAL stages.");

            pdal::Options opts;

            opts.add("filename", input_file);
            reader->setOptions(opts);

            std::ostringstream bounds_stream;

            bounds_stream << std::setprecision(std::numeric_limits<double>::max_digits10)
                          << "([" << b.xmin << "," << b.xmax
                          << "],[" << b.ymin << "," << b.ymax << "])";

            opts = pdal::Options();
            std::cout << "Coordinates " + bounds_stream.str() + "\n";
            opts.add("bounds", bounds_stream.str());
            crop->setOptions(opts);
            crop->setInput(*reader);

            opts = pdal::Options();
            opts.add("value", "QuadratID = " + std::to_string(quadrat_id));
            assign_id->setOptions(opts);
            assign_id->setInput(*crop);

            opts = pdal::Options();
            opts.add("value", "QuadratPosition = " + std::to_string(position_code));
            assign_pos->setOptions(opts);
            assign_pos->setInput(*assign_id);

            opts = pdal::Options();
            opts.add("forward", "scale,offset");
            opts.add("filename", output_path.string());
            opts.add("extra_dims", "QuadratID=int32,QuadratPosition=uint8");

            writer->setOptions(opts);
            writer->setInput(*assign_pos);

            pdal::PointTable table;
            writer->prepare(table);

            const auto views = writer->execute(table);

            std::size_t point_count = 0;
            for (const auto& view : views)
                point_count += view->size();

            if (point_count == 0)
                throw std::runtime_error(
                        "No points found inside quadrat " + direction + "."
                );

            output_files.push_back(output_path.string());
        }

        return output_files;
    }
}

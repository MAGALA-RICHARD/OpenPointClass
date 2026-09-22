// Extract OpenPointClass multiscale features while preserving selected
// point-level scalar fields from PDAL-supported point clouds.
//
// Output contains one row per original point. OpenPointClass features are
// calculated on its base voxel cloud and attached to each original point via
// PointSet::pointMap. This preserves fields such as PlotID without averaging.

#include "constants.hpp"
#include "features.hpp"
#include "labels.hpp"
#include "point_io.hpp"
#include "scale.hpp"
#include "vendor/cxxopts.hpp"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace fs = std::filesystem;

namespace {

struct FeatureDeleter {
  void operator()(Feature* feature) const noexcept { delete feature; }
};

struct ScaleDeleter {
  void operator()(Scale* scale) const noexcept { delete scale; }
};

struct PointSetDeleter {
  void operator()(PointSet* point_set) const noexcept {
    if (point_set != nullptr) {
      point_set->freeIndex<KdTree>();
      delete point_set;
    }
  }
};

using FeaturePtr = std::unique_ptr<Feature, FeatureDeleter>;
using ScalePtr = std::unique_ptr<Scale, ScaleDeleter>;
using PointSetPtr = std::unique_ptr<PointSet, PointSetDeleter>;

std::string csvField(const std::string& value) {
  if (value.find_first_of(",\"\n\r") == std::string::npos) {
    return value;
  }

  std::string escaped = "\"";
  for (const char c : value) {
    escaped += (c == '"') ? "\"\"" : std::string(1, c);
  }
  escaped += '"';
  return escaped;
}

std::string lowerCopy(std::string value) {
  std::transform(
      value.begin(), value.end(), value.begin(),
      [](const unsigned char c) {
        return static_cast<char>(std::tolower(c));
      }
  );
  return value;
}

fs::path outputPathFor(
    const fs::path& requested_output,
    const fs::path& input,
    const bool multiple_inputs
) {
  if (!multiple_inputs) {
    return requested_output;
  }

  if (requested_output.has_extension()) {
    throw std::invalid_argument(
        "--output must be a directory when multiple inputs are supplied"
    );
  }

  fs::create_directories(requested_output);
  return requested_output / (input.stem().string() + "_features.csv");
}

std::vector<std::string> uniqueFeatureNames(
    const std::vector<FeaturePtr>& features
) {
  std::unordered_map<std::string, std::size_t> counts;
  std::vector<std::string> names;
  names.reserve(features.size());

  for (const auto& feature : features) {
    const std::string base_name = feature->getName();
    const std::size_t occurrence = ++counts[base_name];
    names.push_back(
        occurrence == 1
            ? base_name
            : base_name + "__" + std::to_string(occurrence)
    );
  }
  return names;
}

int externalClassification(const PointSet& source, const std::size_t point_id) {
  if (!source.hasLabels()) {
    return -1;
  }

  const int training_label = source.labels[point_id];
  if (training_label == LABEL_UNASSIGNED) {
    return -1;
  }

  const auto training_to_external = getTrain2AsprsCodes();
  const auto mapped = training_to_external.find(training_label);
  return mapped == training_to_external.end()
             ? training_label
             : mapped->second;
}

#ifdef WITH_PDAL

struct SelectedDimension {
  pdal::Dimension::Id id;
  std::string input_name;
  std::string output_name;
};

std::vector<std::pair<pdal::Dimension::Id, std::string>> availableDimensions(
    const PointSet& source
) {
  std::vector<std::pair<pdal::Dimension::Id, std::string>> dimensions;
  if (!source.pointView) {
    return dimensions;
  }

  for (const auto dimension_id : source.pointView->dims()) {
    dimensions.emplace_back(
        dimension_id,
        source.pointView->dimName(dimension_id)
    );
  }

  std::sort(
      dimensions.begin(), dimensions.end(),
      [](const auto& left, const auto& right) {
        return lowerCopy(left.second) < lowerCopy(right.second);
      }
  );
  return dimensions;
}

void printAvailableDimensions(const PointSet& source, const fs::path& input) {
  const auto dimensions = availableDimensions(source);
  std::cout << "Available dimensions in " << input << ":" << std::endl;

  if (dimensions.empty()) {
    std::cout << "  (none exposed by PDAL)" << std::endl;
    return;
  }

  for (const auto& dimension : dimensions) {
    std::cout << "  " << dimension.second << std::endl;
  }
}

std::string availableDimensionMessage(const PointSet& source) {
  std::ostringstream message;
  message << "Available dimensions:";
  for (const auto& dimension : availableDimensions(source)) {
    message << " " << dimension.second;
  }
  return message.str();
}

bool conflictsWithCoreColumn(const std::string& name) {
  const std::string lower_name = lowerCopy(name);
  return lower_name == "source" || lower_name == "point_id" ||
         lower_name == "voxel_id" || lower_name == "x" ||
         lower_name == "y" || lower_name == "z" ||
         lower_name == "classification";
}

std::vector<SelectedDimension> selectDimensions(
    const PointSet& source,
    const std::vector<std::string>& requested_names
) {
  std::vector<SelectedDimension> selected;
  if (requested_names.empty()) {
    return selected;
  }

  if (!source.pointView) {
    throw std::runtime_error(
        "Input scalar fields require a PDAL-supported input such as LAS/LAZ"
    );
  }

  const auto available = availableDimensions(source);
  std::unordered_map<std::string, bool> already_selected;

  for (const auto& requested_name : requested_names) {
    const std::string requested_lower = lowerCopy(requested_name);
    const auto match = std::find_if(
        available.begin(), available.end(),
        [&requested_lower](const auto& dimension) {
          return lowerCopy(dimension.second) == requested_lower;
        }
    );

    if (match == available.end()) {
      throw std::runtime_error(
          "Requested field '" + requested_name +
          "' was not found. " + availableDimensionMessage(source)
      );
    }

    const std::string canonical_lower = lowerCopy(match->second);
    if (already_selected[canonical_lower]) {
      continue;
    }
    already_selected[canonical_lower] = true;

    const std::string output_name = conflictsWithCoreColumn(match->second)
                                        ? "input_" + match->second
                                        : match->second;

    selected.push_back({match->first, match->second, output_name});
  }
  return selected;
}

#else

void printAvailableDimensions(const PointSet&, const fs::path& input) {
  std::cout << "No PDAL dimensions are available for " << input
            << " because OpenPointClass was built without PDAL."
            << std::endl;
}

#endif

void writeCsv(
    const fs::path& output,
    const fs::path& input,
    const PointSet& source,
    const std::vector<FeaturePtr>& features,
    const std::vector<std::string>& requested_fields
) {
  if (output.has_parent_path()) {
    fs::create_directories(output.parent_path());
  }

#ifdef WITH_PDAL
  const auto selected_dimensions = selectDimensions(source, requested_fields);
#else
  if (!requested_fields.empty()) {
    throw std::runtime_error(
        "--fields requires OpenPointClass to be built with PDAL"
    );
  }
#endif

  std::ofstream stream(output);
  if (!stream) {
    throw std::runtime_error("Cannot open output file: " + output.string());
  }

  const auto feature_names = uniqueFeatureNames(features);
  stream << "source,point_id,voxel_id,x,y,z,classification";

#ifdef WITH_PDAL
  for (const auto& dimension : selected_dimensions) {
    stream << ',' << csvField(dimension.output_name);
  }
#endif

  for (const auto& feature_name : feature_names) {
    stream << ',' << csvField(feature_name);
  }
  stream << '\n';
  stream << std::setprecision(std::numeric_limits<double>::max_digits10);

  // One output row per original point preserves the requested fields. The
  // multiscale features are attached from that point's base voxel.
  for (std::size_t point_id = 0; point_id < source.count(); ++point_id) {
    if (point_id >= source.pointMap.size()) {
      throw std::runtime_error(
          "OpenPointClass did not create a voxel mapping for every point"
      );
    }

    const std::size_t voxel_id = source.pointMap[point_id];
    if (voxel_id >= source.base->count()) {
      throw std::runtime_error("Invalid voxel ID in OpenPointClass point map");
    }

    stream << csvField(input.filename().string())
           << ',' << point_id
           << ',' << voxel_id
           << ',' << source.points[point_id][0]
           << ',' << source.points[point_id][1]
           << ',' << source.points[point_id][2]
           << ',';

    const int classification = externalClassification(source, point_id);
    if (classification >= 0) {
      stream << classification;
    }

#ifdef WITH_PDAL
    for (const auto& dimension : selected_dimensions) {
      stream << ',';
      const double value = source.pointView->getFieldAs<double>(
          dimension.id,
          static_cast<pdal::PointId>(point_id)
      );
      if (std::isfinite(value)) {
        stream << value;
      }
    }
#endif

    for (const auto& feature : features) {
      stream << ',' << feature->getValue(voxel_id);
    }
    stream << '\n';

    if (!stream) {
      throw std::runtime_error("Failed while writing: " + output.string());
    }
  }
}

void processOne(
    const fs::path& input,
    const fs::path& output,
    double& start_resolution,
    const int number_of_scales,
    const double radius,
    const std::vector<std::string>& requested_fields,
    const bool list_fields
) {
  std::cout << "Reading " << input << std::endl;
  PointSetPtr point_set(readPointSet(input.string()));

  if (point_set->count() == 0) {
    throw std::runtime_error("Input contains no points: " + input.string());
  }

  if (list_fields) {
    printAvailableDimensions(*point_set, input);
    return;
  }

#ifdef WITH_PDAL
  // Validate requested fields before doing expensive feature calculations.
  selectDimensions(*point_set, requested_fields);
#else
  if (!requested_fields.empty()) {
    throw std::runtime_error(
        "--fields requires OpenPointClass to be built with PDAL"
    );
  }
#endif

  if (start_resolution < 0.0) {
    start_resolution = point_set->spacing();
    std::cout << "Estimated starting resolution: "
              << start_resolution << " m" << std::endl;
  }

  auto raw_scales = computeScales(
      static_cast<std::size_t>(number_of_scales),
      point_set.get(), start_resolution, radius
  );

  std::vector<ScalePtr> scales;
  scales.reserve(raw_scales.size());
  for (auto* scale : raw_scales) {
    scales.emplace_back(scale);
  }

  auto raw_features = getFeatures(raw_scales);
  std::vector<FeaturePtr> features;
  features.reserve(raw_features.size());
  for (auto* feature : raw_features) {
    features.emplace_back(feature);
  }

  std::cout << "Writing " << point_set->count()
            << " original points with " << features.size()
            << " OpenPointClass features to " << output << std::endl;

  writeCsv(output, input, *point_set, features, requested_fields);
}

}  // namespace

int main(int argc, char** argv) {
  cxxopts::Options options(
      "pcdata",
      "Extract OpenPointClass features and selected input scalar fields"
  );

  options.add_options()
      ("i,input", "Input point cloud(s)",
       cxxopts::value<std::vector<std::string>>())
          ("o,output", "Output CSV, or directory for multiple inputs",
           cxxopts::value<std::string>()->default_value("features.csv"))
              ("r,resolution", "First-scale resolution in metres (-1 = estimate)",
               cxxopts::value<double>()->default_value("-1"))
                  ("s,scales", "Number of scales",
                   cxxopts::value<int>()->default_value(MKSTR(NUM_SCALES)))
                      ("radius", "Neighbourhood colour-search radius in metres",
                       cxxopts::value<double>()->default_value(MKSTR(RADIUS)))
                          ("fields", "Input fields to preserve, e.g. PlotID,Biomass",
                           cxxopts::value<std::vector<std::string>>())
                              ("list-fields", "List fields detected by PDAL and exit",
                               cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
                                  ("h,help", "Print usage");

  options.parse_positional({"input"});
  options.positional_help("[point cloud(s)]");

  try {
    const auto result = options.parse(argc, argv);
    if (result.count("help") || !result.count("input")) {
      std::cout << options.help() << std::endl;
      return result.count("help") ? EXIT_SUCCESS : EXIT_FAILURE;
    }

    const auto inputs = result["input"].as<std::vector<std::string>>();
    const fs::path requested_output(result["output"].as<std::string>());
    double resolution = result["resolution"].as<double>();
    const int number_of_scales = result["scales"].as<int>();
    const double radius = result["radius"].as<double>();
    const bool list_fields = result["list-fields"].as<bool>();

    std::vector<std::string> requested_fields;
    if (result.count("fields")) {
      requested_fields =
          result["fields"].as<std::vector<std::string>>();
    }

    if (number_of_scales < 1) {
      throw std::invalid_argument("--scales must be at least 1");
    }
    if (resolution <= 0.0 && resolution != -1.0) {
      throw std::invalid_argument("--resolution must be positive or -1");
    }
    if (radius <= 0.0) {
      throw std::invalid_argument("--radius must be greater than 0");
    }

    const bool multiple_inputs = inputs.size() > 1;
    for (const auto& filename : inputs) {
      const fs::path input(filename);
      const fs::path output = outputPathFor(
          requested_output, input, multiple_inputs
      );

      processOne(
          input, output, resolution, number_of_scales, radius,
          requested_fields, list_fields
      );
    }
  }
  catch (const std::exception& error) {
    std::cerr << "pcdata error: " << error.what() << std::endl;
    return EXIT_FAILURE;
  }

  return EXIT_SUCCESS;
}

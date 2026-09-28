/*
pcdata C++17 rewrite — integration notes
--------------------------------------
* Uses the existing OpenPointClass headers, ownership conventions, computeScales
  and getFeatures. Replace only the pcdata translation unit in your project.
* --hag=false bypasses all HAG allocation/classification/interpolation.
* --hag-existing-ground (with --hag) uses original ASPRS Classification=2.
* --hag-extrapolate defaults false (changed from the earlier implementation).
* hag_status is a diagnostic, NOT LAS Classification (ground remains 2).
* hag_status: 0 ground, 1 finite nonzero estimate, 2 non-ground zero (ambiguous),
  3 nonfinite. Status 1 is not proof of adequate ground support. Unlimited search
  can still interpolate from distant ground points. Validate on surveyed ground.
* CSV keeps original OPC coordinates and labels. PDAL HAG uses original PDAL XYZ;
  the reader must maintain PointSet/pointView row correspondence. This rewrite
  preserves IDs through the HAG stages, but cannot verify the absent reader code.
* No outlier removal, clipping of negative HAG, or change to OPC feature formulas.
* CSV numbers use shortest round-trip representations. Integer fields remain
  exact. An all-zero numeric column may be inferred as integer by pandas; use
  dtype={"height_above_ground": float} if a float dtype is required.
* Output must not already exist. Failed writes leave a .partial file for diagnosis;
  remove/rename it before retrying. Do not run concurrent jobs to the same output.
* Build with C++17 and a library supporting floating-point std::to_chars.
* Full integration/build and measured speedup require your OPC headers and data.
*/
// Extract OpenPointClass multiscale features while preserving selected
// point-level scalar fields from PDAL-supported point clouds.
//
// OpenPointClass features are calculated only through computeScales() and
// getFeatures(). pcdata does not recalculate or alter OpenPointClass features.
//
// HeightAboveGround (HAG) is calculated ONCE, independently of the
// OpenPointClass resolution/scales, then written as one additional feature.

#include "constants.hpp"
#include "features.hpp"
#include "labels.hpp"
#include "point_io.hpp"
#include "scale.hpp"
#include "vendor/cxxopts.hpp"

#include <algorithm>
#include <chrono>
#include <charconv>
#include <locale>
#include <array>
#include <type_traits>
#include <cctype>
#include <cmath>
#include <cstdint>
#include <cstdlib>
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
#include <unordered_set>
#include <vector>

#ifdef WITH_PDAL
#include <pdal/Dimension.hpp>
#include <pdal/Options.hpp>
#include <pdal/PointTable.hpp>
#include <pdal/PointView.hpp>
#include <pdal/StageFactory.hpp>
#include <pdal/Stage.hpp>
#include <pdal/PointLayout.hpp>
#include <pdal/io/BufferReader.hpp>
#include "helper.hpp"
#endif

namespace fs = std::filesystem;

namespace {

// Build as C++17. Coordinates and all distance parameters must use the same
// linear units (metres if using the CLI defaults). No outlier removal is applied.
// Integration contract: PointSet::pointView and points have identical row order,
// as already required by the original scalar-field export. Verify in point_io.
    class Timer {
        std::string name;
        std::chrono::steady_clock::time_point start = std::chrono::steady_clock::now();
    public:
        explicit Timer(std::string n) : name(std::move(n)) {
            std::cerr << "[start] " << name << '\n';
        }
        ~Timer() {
            std::cerr << "[time] " << name << ": "
                      << std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count()
                      << " s\n";
        }
    };

    struct FeatureDeleter { void operator()(Feature* p) const noexcept { delete p; } };
    struct ScaleDeleter { void operator()(Scale* p) const noexcept { delete p; } };
    struct PointSetDeleter {
        void operator()(PointSet* p) const noexcept {
            if (!p) return;
            p->freeIndex<KdTree>();
            delete p;
        }
    };

    using FeaturePtr = std::unique_ptr<Feature, FeatureDeleter>;
    using ScalePtr = std::unique_ptr<Scale, ScaleDeleter>;
    using PointSetPtr = std::unique_ptr<PointSet, PointSetDeleter>;

    struct HagOptions {
        bool enabled = false;

        // Independent from OpenPointClass --resolution and --scales.
        // This is the ground-model raster resolution passed to SMRF as "cell".
        double resolution = 0.05;
        bool existingGround = false;
        bool extrapolate = false;

        // HAG-NN interpolation controls. maxDistance <= 0 means no distance cutoff.
        int neighbors = 16;
        double maxDistance = -1.0;

        // Explicit SMRF window in coordinate units; PDAL default is 18.0.
        double smrfScalar = 1.25;
        double smrfSlope = 0.15;
        double smrfThreshold = 0.5;
        double smrfWindow = 18.0;
    };

    struct HagData {
        std::vector<double> values;
        std::vector<unsigned char> status; // 0=ground, 1=estimate, 2=ambiguous non-ground zero, 3=nonfinite
        // One value per original input row; status is not a confidence score.
        std::size_t groundCount = 0;
        std::size_t nonGroundCount = 0;
        std::size_t nonGroundZeroCount = 0;

        bool empty() const { return values.empty(); }
    };

    std::string csvField(const std::string& value) {
        if (value.find_first_of(",\"\n\r") == std::string::npos) return value;
        std::string out = "\"";
        for (char c : value) out += c == '"' ? "\"\"" : std::string(1, c);
        return out + '"';
    }

    std::string lowerCopy(std::string value) {
        std::transform(value.begin(), value.end(), value.begin(),
                       [](unsigned char c) { return static_cast<char>(std::tolower(c)); });
        return value;
    }

    fs::path outputPathFor(const fs::path& requested, const fs::path& input, bool multiple) {
        if (!multiple) return requested;
        if (fs::exists(requested) && !fs::is_directory(requested))
            throw std::invalid_argument("--output must be a directory when multiple inputs are supplied");
        fs::create_directories(requested);
        return requested / (input.stem().string() + "_features.csv");
    }

    std::vector<std::string> uniqueFeatureNames(const std::vector<FeaturePtr>& features) {
        std::unordered_map<std::string, std::size_t> counts;
        std::vector<std::string> names;
        names.reserve(features.size());

        for (const auto& feature : features) {
            const std::string name = feature->getName();
            const std::size_t n = ++counts[name];
            names.push_back(n == 1 ? name : name + "__" + std::to_string(n));
        }
        return names;
    }

    int externalClassification(const PointSet& source, std::size_t pointId) {
        if (!source.hasLabels()) return -1;
        const int label = source.labels[pointId];
        if (label == LABEL_UNASSIGNED) return -1;
        static const auto map = getTrain2AsprsCodes();
        const auto it = map.find(label);
        return it == map.end() ? label : it->second;
    }

#ifdef WITH_PDAL

    struct SelectedDimension {
  pdal::Dimension::Id id;
  std::string inputName;
  std::string outputName;
};

std::vector<std::pair<pdal::Dimension::Id, std::string>> availableDimensions(const PointSet& source) {
  std::vector<std::pair<pdal::Dimension::Id, std::string>> dims;
  if (!source.pointView) return dims;

  for (auto id : source.pointView->dims())
    dims.emplace_back(id, source.pointView->dimName(id));

  std::sort(dims.begin(), dims.end(), [](const auto& a, const auto& b) {
    return lowerCopy(a.second) < lowerCopy(b.second);
  });
  return dims;
}

void printAvailableDimensions(const PointSet& source, const fs::path& input) {
  const auto dims = availableDimensions(source);
  std::cout << "Available dimensions in " << input << ":\n";

  if (dims.empty()) {
    std::cout << "  (none exposed by PDAL)\n";
    return;
  }
  for (const auto& dim : dims) std::cout << "  " << dim.second << '\n';
}

std::string availableDimensionMessage(const PointSet& source) {
  std::ostringstream out;
  out << "Available dimensions:";
  for (const auto& dim : availableDimensions(source)) out << ' ' << dim.second;
  return out.str();
}

bool conflictsWithCoreColumn(const std::string& name) {
  const std::string n = lowerCopy(name);
  return n == "source" || n == "point_id" || n == "voxel_id" || n == "x" ||
         n == "y" || n == "z" || n == "classification" || n == "height_above_ground" || n == "hag_status";
}

std::vector<SelectedDimension> selectDimensions(const PointSet& source,
                                                const std::vector<std::string>& requested) {
  std::vector<SelectedDimension> selected;
  if (requested.empty()) return selected;
  if (!source.pointView)
    throw std::runtime_error("Input scalar fields require a PDAL-supported input such as LAS/LAZ");

  const auto available = availableDimensions(source);
  std::unordered_set<std::string> seen;

  for (const auto& requestedName : requested) {
    const std::string wanted = lowerCopy(requestedName);
    const auto it = std::find_if(available.begin(), available.end(), [&](const auto& dim) {
      return lowerCopy(dim.second) == wanted;
    });

    if (it == available.end())
      throw std::runtime_error("Requested field '" + requestedName +
                               "' was not found. " + availableDimensionMessage(source));

    const std::string key = lowerCopy(it->second);
    if (!seen.insert(key).second) continue;

    selected.push_back({it->first, it->second,
                        conflictsWithCoreColumn(it->second)
                            ? "input_" + it->second
                            : it->second});
  }
  return selected;
}

HagData computeHag(const PointSet& source, const HagOptions& cfg) {
  using namespace pdal;
  Timer total("HAG total");
  if (!source.pointView || source.pointView->size() != source.count())
    throw std::runtime_error("HAG requires a PDAL view matching the original point rows");
  const auto& original = *source.pointView;
  const std::size_t n = source.count();

  // Independent table: no shared point storage with the classification dataset.
  PointTable table;
  auto layout = table.layout();
  layout->registerDim(Dimension::Id::X);
  layout->registerDim(Dimension::Id::Y);
  layout->registerDim(Dimension::Id::Z);
  layout->registerDim(Dimension::Id::Classification);
  // hag_nn registers its required HAG dimension/type during prepare().
  layout->registerDim(Dimension::Id::ReturnNumber);
  layout->registerDim(Dimension::Id::NumberOfReturns);
  const auto rowId = layout->registerOrAssignDim("PcdataOriginalRow", Dimension::Type::Unsigned64);
  // Prepare the entire pipeline before adding any points. A downstream stage
  // may register a dimension or widen its type during prepare().
  StageFactory factory;
  BufferReader buffer;
  Stage* inputStage = &buffer;
  if (!cfg.existingGround) {
    Stage* smrf = factory.createStage("filters.smrf");
    if (!smrf) throw std::runtime_error("PDAL filters.smrf unavailable");
    Options opts;
    opts.add("cell",cfg.resolution);
    opts.add("scalar",cfg.smrfScalar);
    opts.add("slope",cfg.smrfSlope);
    opts.add("threshold",cfg.smrfThreshold);
    opts.add("window",cfg.smrfWindow);
    opts.add("returns","first,intermediate,last,only");
    smrf->setOptions(opts);
    smrf->setInput(buffer);
    inputStage=smrf;
  }
  Stage* hag = factory.createStage("filters.hag_nn");
  if (!hag) throw std::runtime_error("PDAL filters.hag_nn unavailable");
  Options opts;
  opts.add("count",cfg.neighbors);
  opts.add("allow_extrapolation",cfg.extrapolate);
  if (cfg.maxDistance>0) opts.add("max_distance",cfg.maxDistance);
  hag->setOptions(opts);
  hag->setInput(*inputStage);
  {
    Timer timer("prepare complete HAG pipeline");
    hag->prepare(table);
  }
  auto working = std::make_shared<PointView>(table);
  {
    Timer timer("HAG coordinate copy");
    if (cfg.existingGround && !original.hasDim(Dimension::Id::Classification))
      throw std::runtime_error("--hag-existing-ground requires Classification");
    for (PointId i=0; i<n; ++i) {
      for (auto dim : {Dimension::Id::X, Dimension::Id::Y, Dimension::Id::Z}) {
        const double value = original.getFieldAs<double>(dim,i);
        if (!std::isfinite(value)) throw std::runtime_error("Nonfinite input coordinate");
        working->setField(dim,i,value);
      }
      working->setField(rowId,i,static_cast<std::uint64_t>(i));
      // Preserve valid return metadata; photogrammetry often supplies 0/0.
      auto rn=original.hasDim(Dimension::Id::ReturnNumber)
        ? original.getFieldAs<unsigned>(Dimension::Id::ReturnNumber,i) : 0;
      auto nr=original.hasDim(Dimension::Id::NumberOfReturns)
        ? original.getFieldAs<unsigned>(Dimension::Id::NumberOfReturns,i) : 0;
      if (!rn && !nr) rn=nr=1;
      if (!rn || rn>nr || nr>255) throw std::runtime_error("Inconsistent return metadata");
      working->setField(Dimension::Id::ReturnNumber,i,rn);
      working->setField(Dimension::Id::NumberOfReturns,i,nr);
      // Reset only the working labels when fitting a new ground model.
      working->setField(Dimension::Id::Classification,i,
        cfg.existingGround ? original.getFieldAs<unsigned char>(Dimension::Id::Classification,i) : 0);
    }
  }
  buffer.addView(working);
  if (cfg.existingGround) {
    std::size_t groundCount=0;
    for (PointId i=0;i<n;++i)
      groundCount += working->getFieldAs<unsigned char>(Dimension::Id::Classification,i)==2;
    if (!groundCount) throw std::runtime_error("No Classification=2 ground points");
  }
  PointViewSet views;
  {
    Timer timer(cfg.existingGround ? "HAG interpolation" : "SMRF and HAG interpolation");
    views=hag->execute(table);
  }
  if (views.size()!=1 || (*views.begin())->size()!=n)
    throw std::runtime_error("HAG changed the view count or point count");
  const auto view=*views.begin();
  HagData out;
  out.values.resize(n,std::numeric_limits<double>::quiet_NaN());
  out.status.resize(n,3);
  std::vector<bool> seen(n,false);
  for (PointId i=0;i<n;++i) {
    const auto id=view->getFieldAs<std::uint64_t>(rowId,i);
    if (id>=n || seen[id]) throw std::runtime_error("Invalid or duplicate original point ID");
    seen[id]=true;
    const bool ground=view->getFieldAs<unsigned char>(Dimension::Id::Classification,i)==2;
    const double value=view->getFieldAs<double>(Dimension::Id::HeightAboveGround,i);
    out.values[id]=value; // Keep signed values; never clip negative heights.
    if (ground) ++out.groundCount; else ++out.nonGroundCount;
    if (!std::isfinite(value)) out.status[id]=3;
    else if (ground) out.status[id]=0;
    else if (value==0.0) { out.status[id]=2; ++out.nonGroundZeroCount; }
    else out.status[id]=1;
  }
  std::cerr << "HAG ground=" << out.groundCount << ", non-ground=" << out.nonGroundCount
    << ", ambiguous non-ground zeros=" << out.nonGroundZeroCount << '\n';
  if (out.nonGroundZeroCount)
    std::cerr << "Non-ground zeros may be genuine or PDAL interpolation fallbacks; inspect hag_status.\n";
  if (out.groundCount > 0.95*n)
    std::cerr << "More than 95% ground: inspect the ground model against independent observations.\n";
  return out;
}

#else

    void printAvailableDimensions(const PointSet&, const fs::path& input) {
        std::cout << "No PDAL dimensions are available for " << input
                  << " because OpenPointClass was built without PDAL.\n";
    }

#endif

// Locale-independent, buffered numeric CSV output (including exact integer IDs).
    class CsvWriter {
        std::ofstream stream;
        std::string buffer;
    public:
        explicit CsvWriter(const fs::path& path) : stream(path,std::ios::binary|std::ios::trunc) {
            if (!stream) throw std::runtime_error("Cannot open " + path.string());
            stream.exceptions(std::ios::badbit|std::ios::failbit);
            buffer.reserve(1<<20);
        }
        CsvWriter& operator<<(const std::string& value) { buffer+=value; return *this; }
        CsvWriter& operator<<(const char* value) { buffer+=value; return *this; }
        CsvWriter& operator<<(char value) {
            buffer+=value;
            if (value=='\n' && buffer.size()>=(1<<20)) flush();
            return *this;
        }
        template<class T> CsvWriter& operator<<(T value) {
            if constexpr (std::is_floating_point_v<T>) if (!std::isfinite(value)) return *this;
            std::array<char,128> text;
            auto result=std::to_chars(text.data(),text.data()+text.size(),value);
            if (result.ec!=std::errc{}) throw std::runtime_error("Numeric CSV conversion failed");
            buffer.append(text.data(),result.ptr);
            return *this;
        }
        void flush() { stream.write(buffer.data(),static_cast<std::streamsize>(buffer.size())); buffer.clear(); }
        void close() { flush(); stream.flush(); stream.close(); }
    };

    void writeCsv(const fs::path& output, const fs::path& input, const PointSet& source,
                  const std::vector<FeaturePtr>& features,
                  const std::vector<std::string>& requestedFields, const HagData& hag) {
        if (output.has_parent_path()) fs::create_directories(output.parent_path());

#ifdef WITH_PDAL
        const auto selected = selectDimensions(source, requestedFields);
#else
        if (!requestedFields.empty())
            throw std::runtime_error("--fields requires OpenPointClass to be built with PDAL");
#endif

        if (fs::exists(output)) throw std::runtime_error("Output already exists: " + output.string());
        if (source.pointMap.size()!=source.count() || !source.base)
            throw std::runtime_error("Incomplete OpenPointClass voxel mapping");
        if (!hag.empty() && (hag.values.size()!=source.count() || hag.status.size()!=source.count()))
            throw std::runtime_error("Incomplete HAG mapping");
        const fs::path temporary=output.string()+".partial";
        if (fs::exists(temporary)) throw std::runtime_error("Partial output already exists: " + temporary.string());
        const std::string sourceName=csvField(input.filename().string());

        const auto featureNames = uniqueFeatureNames(features);
        std::unordered_set<std::string> columns={"source","point_id","voxel_id","x","y","z","classification"};
        auto reserveColumn=[&](const std::string& name) {
            if (!columns.insert(lowerCopy(name)).second) throw std::runtime_error("Duplicate CSV column: "+name);
        };
#ifdef WITH_PDAL
        for (const auto& dim:selected) reserveColumn(dim.outputName);
#endif
        if (!hag.empty()) { reserveColumn("height_above_ground"); reserveColumn("hag_status"); }
        for (const auto& name:featureNames) reserveColumn(name);

        CsvWriter stream(temporary);
        stream << "source,point_id,voxel_id,x,y,z,classification";
#ifdef WITH_PDAL
        for (const auto& dim : selected) stream << ',' << csvField(dim.outputName);
#endif
        if (!hag.empty()) stream << ",height_above_ground,hag_status";
        for (const auto& name : featureNames) stream << ',' << csvField(name);
        stream << '\n';

        for (std::size_t pointId = 0; pointId < source.count(); ++pointId) {
            if (pointId >= source.pointMap.size())
                throw std::runtime_error("OpenPointClass did not create a voxel mapping for every point");

            const std::size_t voxelId = source.pointMap[pointId];
            if (!source.base || voxelId >= source.base->count())
                throw std::runtime_error("Invalid voxel ID in OpenPointClass point map");

            stream << sourceName << ',' << pointId << ',' << voxelId
                   << ',' << source.points[pointId][0]
                   << ',' << source.points[pointId][1]
                   << ',' << source.points[pointId][2] << ',';

            const int classification = externalClassification(source, pointId);
            if (classification >= 0) stream << classification;

#ifdef WITH_PDAL
            for (const auto& dim : selected) {
      stream << ',';
      const auto type=source.pointView->layout()->dimType(dim.id);
      using Type=pdal::Dimension::Type;
      switch(type) {
        case Type::Unsigned8: case Type::Unsigned16: case Type::Unsigned32: case Type::Unsigned64:
          stream << source.pointView->getFieldAs<std::uint64_t>(dim.id,pointId); break;
        case Type::Signed8: case Type::Signed16: case Type::Signed32: case Type::Signed64:
          stream << source.pointView->getFieldAs<std::int64_t>(dim.id,pointId); break;
        default: stream << source.pointView->getFieldAs<double>(dim.id,pointId); break;
      }
    }
#endif

            if (!hag.empty()) {
                stream << ',';
                stream << hag.values[pointId] << ',' << static_cast<unsigned>(hag.status[pointId]);
            }

            // All existing OpenPointClass features remain exactly as returned by getFeatures().
            for (const auto& feature : features) stream << ',' << feature->getValue(voxelId);

            stream << '\n';

        }
        stream.close();
        fs::rename(temporary,output);
    }

    void processOne(const fs::path& input, const fs::path& output, double requestedResolution,
                    int numScales, double radius, const std::vector<std::string>& requestedFields,
                    bool listFields, const HagOptions& hagOptions) {
        if (!listFields && (fs::exists(output) || fs::exists(output.string()+".partial")))
            throw std::runtime_error("Output or partial output already exists: " + output.string());
        std::cout << "Reading " << input << '\n';

        Timer total("file total");
        PointSetPtr pointSet;
        { Timer timer("load input"); pointSet.reset(readPointSet(input.string())); }
        std::cerr << "HAG enabled=" << std::boolalpha << hagOptions.enabled << '\n';
        if (!pointSet || pointSet->count() == 0)
            throw std::runtime_error("Input contains no points: " + input.string());

        if (listFields) {
            printAvailableDimensions(*pointSet, input);
            return;
        }

#ifdef WITH_PDAL
        if (pointSet->pointView && pointSet->pointView->size()!=pointSet->count())
    throw std::runtime_error("PDAL view and PointSet row counts differ");
  selectDimensions(*pointSet, requestedFields);
#else
        if (!requestedFields.empty())
            throw std::runtime_error("--fields requires OpenPointClass to be built with PDAL");
        if (hagOptions.enabled)
            throw std::runtime_error("--hag requires OpenPointClass to be built with PDAL");
#endif

        double startResolution = requestedResolution;
        if (startResolution < 0.0) {
            Timer timer("estimate spacing");
            startResolution = pointSet->spacing();
            std::cout << "Estimated OpenPointClass starting resolution: "
                      << startResolution << " m\n";
        }

        std::cerr << "Parameters: resolution=" << startResolution << ", scales=" << numScales
                  << ", radius=" << radius << '\n';
        if (hagOptions.enabled)
            std::cerr << "HAG: existing_ground=" << hagOptions.existingGround
                      << ", cell=" << hagOptions.resolution << ", window=" << hagOptions.smrfWindow
                      << ", scalar=" << hagOptions.smrfScalar << ", slope=" << hagOptions.smrfSlope
                      << ", threshold=" << hagOptions.smrfThreshold << ", neighbors=" << hagOptions.neighbors
                      << ", max_distance=" << hagOptions.maxDistance << ", extrapolate=" << hagOptions.extrapolate << '\n';
        HagData hag;
#ifdef WITH_PDAL
        if (hagOptions.enabled) hag = computeHag(*pointSet, hagOptions);
#endif

        // OpenPointClass multiscale feature calculation is unchanged and is completely
        // independent of HAG resolution.
        if (!std::isfinite(startResolution) || startResolution<=0)
            throw std::runtime_error("Invalid estimated resolution");
        auto rawScales = [&] {
            Timer timer("compute scales");
            return computeScales(static_cast<std::size_t>(numScales),pointSet.get(),startResolution,radius);
        }();

        std::vector<ScalePtr> scales;
        scales.reserve(rawScales.size());
        for (auto* scale : rawScales) scales.emplace_back(scale);

        auto rawFeatures = [&] { Timer timer("create features"); return getFeatures(rawScales); }();
        std::vector<FeaturePtr> features;
        features.reserve(rawFeatures.size());
        for (auto* feature : rawFeatures) features.emplace_back(feature);

        std::cout << "Writing " << pointSet->count() << " original points with "
                  << features.size() << " OpenPointClass features";
        if (!hag.empty()) std::cout << " + 1 HAG feature";
        std::cout << " to " << output << '\n';

        { Timer timer("evaluate features and export CSV");
            writeCsv(output, input, *pointSet, features, requestedFields, hag); }
    }

}  // namespace

int main(int argc, char** argv) {
    cxxopts::Options options(
            "pcdata",
            "Extract OpenPointClass multiscale features, selected scalar fields, and optional single-scale HAG");

    options.add_options()
            ("i,input", "Input point cloud(s)", cxxopts::value<std::vector<std::string>>())
            ("o,output", "Output CSV, or directory for multiple inputs",
             cxxopts::value<std::string>()->default_value("features.csv"))
            ("r,resolution", "OpenPointClass first-scale resolution in metres (-1 = estimate)",
             cxxopts::value<double>()->default_value("-1"))
            ("s,scales", "Number of OpenPointClass scales",
             cxxopts::value<int>()->default_value(MKSTR(NUM_SCALES)))
            ("radius", "OpenPointClass radius in metres",
             cxxopts::value<double>()->default_value(MKSTR(RADIUS)))
            ("fields", "Input fields to preserve, e.g. PlotID,Biomass",
             cxxopts::value<std::vector<std::string>>())
            ("list-fields", "List fields detected by PDAL and exit",
             cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("hag", "Compute one HeightAboveGround feature using PDAL SMRF + HAG-NN",
             cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("hag-resolution", "Independent HAG ground-model resolution in metres",
             cxxopts::value<double>()->default_value("0.05"))
            ("hag-neighbors", "Nearest ground points used by HAG-NN",
             cxxopts::value<int>()->default_value("16"))
            ("hag-max-distance", "Maximum HAG ground-search distance in metres (<=0 = no cutoff)",
             cxxopts::value<double>()->default_value("-1"))
            ("smrf-scalar", "PDAL SMRF scalar",
             cxxopts::value<double>()->default_value("1.25"))
            ("smrf-slope", "PDAL SMRF slope",
             cxxopts::value<double>()->default_value("0.15"))
            ("smrf-threshold", "PDAL SMRF elevation threshold in metres",
             cxxopts::value<double>()->default_value("0.5"))
            ("smrf-window", "PDAL SMRF maximum window in coordinate units",
             cxxopts::value<double>()->default_value("18"))
            ("hag-existing-ground", "Use existing Classification=2; skip SMRF", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("hag-extrapolate", "Allow HAG extrapolation outside ground extent", cxxopts::value<bool>()->default_value("false")->implicit_value("true"))
            ("h,help", "Print usage");

    options.parse_positional({"input"});
    options.positional_help("[point cloud(s)]");

    try {
        const auto result = options.parse(argc, argv);
        if (result.count("help") || !result.count("input")) {
            std::cout << options.help() << '\n';
            return result.count("help") ? EXIT_SUCCESS : EXIT_FAILURE;
        }

        const auto inputs = result["input"].as<std::vector<std::string>>();
        const fs::path requestedOutput(result["output"].as<std::string>());
        const double resolution = result["resolution"].as<double>();
        const int numScales = result["scales"].as<int>();
        const double radius = result["radius"].as<double>();
        const bool listFields = result["list-fields"].as<bool>();

        std::vector<std::string> requestedFields;
        if (result.count("fields"))
            requestedFields = result["fields"].as<std::vector<std::string>>();

        HagOptions hag;
        hag.enabled = result["hag"].as<bool>();
        hag.existingGround=result["hag-existing-ground"].as<bool>();
        hag.extrapolate=result["hag-extrapolate"].as<bool>();
        hag.resolution = result["hag-resolution"].as<double>();
        hag.neighbors = result["hag-neighbors"].as<int>();
        hag.maxDistance = result["hag-max-distance"].as<double>();
        hag.smrfScalar = result["smrf-scalar"].as<double>();
        hag.smrfSlope = result["smrf-slope"].as<double>();
        hag.smrfThreshold = result["smrf-threshold"].as<double>();
        hag.smrfWindow = result["smrf-window"].as<double>();

        for (double value : {resolution,radius,hag.resolution,hag.maxDistance,hag.smrfScalar,
                             hag.smrfSlope,hag.smrfThreshold,hag.smrfWindow})
            if (!std::isfinite(value)) throw std::invalid_argument("Numeric options must be finite");
        if (hag.enabled && hag.smrfWindow<=0) throw std::invalid_argument("--smrf-window must be positive");
        if (numScales < 1)
            throw std::invalid_argument("--scales must be at least 1");
        if (resolution <= 0.0 && resolution != -1.0)
            throw std::invalid_argument("--resolution must be positive or -1");
        if (radius <= 0.0)
            throw std::invalid_argument("--radius must be greater than 0");

        if (hag.enabled) {
            if (hag.resolution <= 0.0)
                throw std::invalid_argument("--hag-resolution must be greater than 0");
            if (hag.neighbors < 1)
                throw std::invalid_argument("--hag-neighbors must be at least 1");
            if (hag.smrfScalar <= 0.0)
                throw std::invalid_argument("--smrf-scalar must be greater than 0");
            if (hag.smrfSlope < 0.0)
                throw std::invalid_argument("--smrf-slope must be non-negative");
            if (hag.smrfThreshold < 0.0)
                throw std::invalid_argument("--smrf-threshold must be non-negative");
        }

        const bool multiple = inputs.size() > 1;
        std::unordered_set<std::string> targets;
        if (!listFields) for (const auto& filename:inputs) {
                const auto target=outputPathFor(requestedOutput,fs::path(filename),multiple);
                const auto key=lowerCopy(fs::absolute(target).lexically_normal().string());
                if (!targets.insert(key).second) throw std::runtime_error("Inputs map to duplicate output: "+target.string());
                if (fs::exists(target) || fs::exists(target.string()+".partial"))
                    throw std::runtime_error("Output or partial output already exists: "+target.string());
            }
        for (const auto& filename : inputs) {
            const fs::path input(filename);
            processOne(input, listFields ? fs::path{} : outputPathFor(requestedOutput, input, multiple), resolution,
                       numScales, radius, requestedFields, listFields, hag);
        }
    }
    catch (const std::exception& e) {
        std::cerr << "pcdata error: " << e.what() << '\n';
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}

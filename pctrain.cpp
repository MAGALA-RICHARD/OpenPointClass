#include "constants.hpp"
#include "point_io.hpp"
#include "classifier.hpp"
#include "randomforest.hpp"
#include "model_metadata.hpp"
#include "svm.hpp"

#include "vendor/cxxopts.hpp"

#ifdef WITH_GBT
#include "gbm.hpp"
#endif


int main(int argc, char **argv) {

    cxxopts::Options options(
        "pctrain",
        "Trains a point cloud classification model"
    );

    options.add_options()
        ("i,input",
            "Ground truth labeled point cloud(s) to use for training",
            cxxopts::value<std::vector<std::string>>())

        ("o,output",
            "Output model",
            cxxopts::value<std::string>()
                ->default_value("model.bin"))

        ("r,resolution",
            "Resolution of the first scale (-1 = estimate automatically)",
            cxxopts::value<double>()
                ->default_value("-1"))

        ("s,scales",
            "Number of scales to compute",
            cxxopts::value<int>()
                ->default_value(MKSTR(NUM_SCALES)))

        ("t,trees",
            "Number of trees in the forest",
            cxxopts::value<int>()
                ->default_value(MKSTR(N_TREES)))

        ("d,depth",
            "Maximum depth of trees",
            cxxopts::value<int>()
                ->default_value(MKSTR(MAX_DEPTH)))

        ("m,max-samples",
            "Approximate maximum number of samples for each input point cloud",
            cxxopts::value<int>()
                ->default_value("100000"))

        ("radius",
            "Radius size to use for neighbor search (meters)",
            cxxopts::value<double>()
                ->default_value(MKSTR(RADIUS)))

        ("e,eval",
            "Labeled point cloud to use for model accuracy evaluation",
            cxxopts::value<std::string>()
                ->default_value(""))

        ("eval-result",
            "Path where to store evaluation results (PLY)",
            cxxopts::value<std::string>()
                ->default_value(""))

        ("stats","Path where to store evaluation statistics (JSON)",
            cxxopts::value<std::string>()
                ->default_value(""))
        ("kernel", "SVM kernel type: git",
         cxxopts::value<std::string>()
           ->default_value("linear"))
        ("degree", "Polynomial kernel degree",
         cxxopts::value<int>()
             ->default_value("3"))

        ("g,gamma", "Kernel coefficient for RBF, polynomial, and sigmoid kernels",
         cxxopts::value<double>()
             ->default_value("0.0"))

        ("C", "SVM regularization parameter C",
         cxxopts::value<double>()
             ->default_value("1.0"))

        ("coef0", "Independent term for polynomial and sigmoid kernels",
         cxxopts::value<double>()
             ->default_value("0.0"))

        ("c,classifier","Which classifier type to use (rf = Random Forest, gbt = Gradient Boosted Trees, svm = Support Vector Machines)",cxxopts::value<std::string>()
                ->default_value("rf"))

        ("classes",
            "Train only these classification classes (comma separated IDs)",
            cxxopts::value<std::vector<int>>())

        ("exclude",
        "Exclude selected features from training (comma-separated names). "
        "Available features: omnivariance,eigenentropy,anisotropy,planarity,"
        "linearity,surface_variation,scatter,verticality,"
        "order_1_axis_1,order_1_axis_2,order_2_axis_1,order_2_axis_2,"
        "vertical_range,height_below,height_above,"
        "point_color_0,point_color_1,point_color_2,"
        "neighborhood_colors_0,neighborhood_colors_1,neighborhood_colors_2,"
        "flower_index, green_leaf_index. "
        "If omitted, all features are used.",  cxxopts::value<std::vector<std::string>>())

        ("h,help",
            "Print usage");

    options.parse_positional({ "input" });
    options.positional_help("[labeled point cloud(s)]");

    cxxopts::ParseResult result;

    try {
        result = options.parse(argc, argv);
    }
    catch (const std::exception &e) {
        std::cerr << e.what() << std::endl;
        std::cerr << options.help() << std::endl;
        return EXIT_FAILURE;
    }

    if (result.count("help") || !result.count("input")) {
        std::cout << options.help() << std::endl;
        return EXIT_SUCCESS;
    }

    try {

        const auto filenames = result["input"].as<std::vector<std::string>>();
        const auto modelFilename = result["output"].as<std::string>();
        double startResolution =  result["resolution"].as<double>();
        const auto scales =  result["scales"].as<int>();
        const auto numTrees =  result["trees"].as<int>();
        const auto treeDepth =  result["depth"].as<int>();
        const auto radius = result["radius"].as<double>();
        const auto maxSamples = result["max-samples"].as<int>();
        const auto classifier = result["classifier"].as<std::string>();
        const auto evalResult =  result["eval-result"].as<std::string>();
        const auto statsFile = result["stats"].as<std::string>();
        const auto evalFilename = result["eval"].as<std::string>();
        const auto kernel = svm::kernelFromString(result["kernel"].as<std::string>());
        const auto degree = result["degree"].as<int>();
        const auto gamma = result["gamma"].as<double>();
        const auto C = result["C"].as<double>();
        const auto coef0 = result["coef0"].as<double>();


        std::vector<int> classes = {};

        if (result.count("classes")) {
            classes = result["classes"].as<std::vector<int>>();
        }


        std::vector<std::string> excludedFeatures = {};

        if (result.count("exclude")) {
            excludedFeatures = result["exclude"].as<std::vector<std::string>>();
        }

        std::cout << "pctrain exclusions: "
          << excludedFeatures.size()
          << std::endl;

            for (const auto &f : excludedFeatures) {
                std::cout << "  " << f << std::endl;
            }

        if (classifier != "rf" &&
                classifier != "gbt" &&
                classifier != "svm") {

                std::cout
                    << options.help()
                    << std::endl;

                return EXIT_FAILURE;
            }


#ifndef WITH_GBT
        if (classifier == "gbt") {

            std::cerr
                << "Gradient Boosted Trees support has not been built "
                << "(try building with -DWITH_GBT=ON)"
                << std::endl;

            return EXIT_FAILURE;
        }
#endif


        std::cout
            << "Using "
            << (
                   classifier == "rf" ? "Random Forest" :
                   classifier == "svm" ? "Support Vector Machine" :
                                       "Gradient Boosted Trees"
                   )
            << std::endl;


        // -------------------------
        // TRAIN
        // -------------------------

        if (classifier == "rf") {

            rf::RandomForest *rtrees =
                rf::train(
                    filenames,
                    &startResolution,
                    scales,
                    numTrees,
                    treeDepth,
                    radius,
                    maxSamples,
                    classes,
                    excludedFeatures
                );

            rf::saveForest(
                rtrees,
                modelFilename
            );

            delete rtrees;
        }
        else if (classifier == "svm") {
            svm::SVM model = svm::train(
                filenames,
                &startResolution,
                scales,
                radius,
                maxSamples,
                classes,
                excludedFeatures,
                kernel,
                C,
                gamma,
                degree,
                coef0
            );
            std::cout << "completed with svm " <<std::endl;
            svm::saveSVM(model, modelFilename);

        }

#ifdef WITH_GBT
        else if (classifier == "gbt") {

            gbm::Boosting *booster =
                gbm::train(
                    filenames,
                    &startResolution,
                    scales,
                    numTrees,
                    treeDepth,
                    radius,
                    maxSamples,
                    classes,
                    excludedFeatures
                );

            gbm::saveBooster(
                booster,
                modelFilename
            );
        }
#endif


        // Save feature exclusions independently
        saveExcludedFeatures(modelFilename, excludedFeatures);

        // -------------------------
        // EVALUATE
        // -------------------------

        if (!evalFilename.empty()) {

            std::cout
                << "Evaluating on "
                << evalFilename
                << " ..."
                << std::endl;

            const ClassifierType ctype = fingerprint(modelFilename);

            rf::RandomForest* rtrees = nullptr;
            svm::SVM svmModel;

#ifdef WITH_GBT
            gbm::Boosting* booster = nullptr;
#endif

            if (ctype == RandomForest) {
              rtrees = rf::loadForest(modelFilename);
            }
            else if (ctype == SupportVectorMachine) {
              svmModel = svm::loadSVM(modelFilename);
            }
#ifdef WITH_GBT
            else if (ctype == GradientBoostedTrees) {
              booster = gbm::loadBooster(modelFilename);
            }
#endif
            else {
              throw std::runtime_error("Unsupported classifier model");
            }
            const auto labels = getTrainingLabels();

            const auto evalPointSet = readPointSet(
                    evalFilename
                );


            if (!evalPointSet->hasLabels()) {

                throw std::runtime_error(
                    "Evaluation dataset has no labels"
                );
            }

            const auto evalFeatures = getFeatures( computeScales(
                        scales,
                        evalPointSet,
                        startResolution,
                        radius),
                    excludedFeatures);

            std::cout
                << "Features: "
                << evalFeatures.size()
                << std::endl;
             std::cout
                << "Excluded: "
                <<  excludedFeatures.size()
                << std::endl;



            if (ctype == RandomForest) {
                rf::classify(
                    *evalPointSet,
                    rtrees,
                    evalFeatures,
                    labels,
                    Regularization::None,
                    2.5,
                    true,
                    false,
                    true,
                    {},
                    statsFile
                );
            }

            else if (ctype == SupportVectorMachine) {
                std::cout << "evaluating with svm " <<std::endl;
                auto model = svm::loadSVM(modelFilename);
                svm::classify(
                    *evalPointSet, model, evalFeatures, labels,
                    Regularization::None, 2.5, true, false, true, {}, statsFile
                );
            }

#ifdef WITH_GBT
            else {

                gbm::classify(
                    *evalPointSet,
                    booster,
                    evalFeatures,
                    labels,
                    Regularization::None,
                    2.5,
                    true,
                    false,
                    true,
                    {},
                    statsFile
                );
            }
#endif
            if (!evalResult.empty()) {
                savePointSet(
                    *evalPointSet,
                    evalResult
                );
            }
        }
    }
    catch (std::exception &e) {
        std::cerr
            << "Error: "
            << e.what()
            << std::endl;

        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
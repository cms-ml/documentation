/*
 * Example plugin to demonstrate the inference with TensorFlow AOT.
 */

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlowAOT/interface/Model.h"

// include the header of the compiled model
#include "tfaot-model-test/model.h"

class MyPlugin : public edm::stream::EDAnalyzer<> {
public:
  explicit MyPlugin(const edm::ParameterSet&);
  ~MyPlugin(){};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void beginJob(){};
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob(){};

  std::vector<std::string> batchRuleStrings_;

  // aot model
  tfaot::Model<tfaot_model::test> model_;
};

void MyPlugin::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // defining this function will lead to a *_cfi file being generated when compiling
  edm::ParameterSetDescription desc;
  desc.add<std::vector<std::string>>("batchRules");
  descriptions.addWithDefaultLabel(desc);
}

MyPlugin::MyPlugin(const edm::ParameterSet& config)
    : batchRuleStrings_(config.getParameter<std::vector<std::string>>("batchRules")) {
  // register batch rules
  for (const auto& rule : batchRuleStrings_) {
    model_.setBatchRule(rule);
  }
}

void MyPlugin::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // define input for a batch size of 1
  // (just a single float input, with shape 1x4)
  tfaot::FloatArrays input = { {0, 1, 2, 3} };

  // define output
  // (just a single float output, which will shape 1x2)
  tfaot::FloatArrays output;

  // evaluate the model
  // the template arguments of run() correspond to the types of the outputs
  // that are "tied" the "1" denote the batch size of 1
  std::tie(output) = model_.run<tfaot::FloatArrays>(1, input);

  // print output
  std::cout << "output[0]: " << output[0][0] << ", " << output[0][1] << std::endl;
  // -> "output[0]: 0.648093, 0.351907"
}

DEFINE_FWK_MODULE(MyPlugin);

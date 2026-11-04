/*
 * Example plugin to demonstrate the direct multi-threaded inference with ONNX Runtime.
 */

#include <memory>
#include <iostream>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"

#include "PhysicsTools/ONNXRuntime/interface/ONNXRuntime.h"

using namespace cms::Ort;

class MyPlugin : public edm::stream::EDAnalyzer<edm::GlobalCache<ONNXRuntime>> {
public:
  explicit MyPlugin(const edm::ParameterSet &, const ONNXRuntime *);
  static void fillDescriptions(edm::ConfigurationDescriptions&);

  static std::unique_ptr<ONNXRuntime> initializeGlobalCache(const edm::ParameterSet &);
  static void globalEndJob(const ONNXRuntime *);

private:
  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

  std::vector<std::string> input_names_;
  std::vector<std::vector<int64_t>> input_shapes_;
  FloatArrays data_; // each stream hosts its own data
};


void MyPlugin::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // defining this function will lead to a *_cfi file being generated when compiling
  edm::ParameterSetDescription desc;
  desc.add<edm::FileInPath>("model_path", edm::FileInPath("MySubsystem/MyModule/data/model.onnx"));
  desc.add<std::vector<std::string>>("input_names", std::vector<std::string>({"my_input"}));
  descriptions.addWithDefaultLabel(desc);
}


MyPlugin::MyPlugin(const edm::ParameterSet &iConfig, const ONNXRuntime *cache)
    : input_names_(iConfig.getParameter<std::vector<std::string>>("input_names")),
      input_shapes_() {
    // initialize the input data arrays
    // note there is only one element in the FloatArrays type (i.e. vector<vector<float>>) variable
    data_.emplace_back(10, 0);
}


std::unique_ptr<ONNXRuntime> MyPlugin::initializeGlobalCache(const edm::ParameterSet &iConfig) {
  return std::make_unique<ONNXRuntime>(iConfig.getParameter<edm::FileInPath>("model_path").fullPath());
}

void MyPlugin::globalEndJob(const ONNXRuntime *cache) {}

void MyPlugin::analyze(const edm::Event &iEvent, const edm::EventSetup &iSetup) {
  // prepare dummy inputs for every event
  std::vector<float> &group_data = data_[0];
  for (size_t i = 0; i < 10; i++){
      group_data[i] = float(iEvent.id().event() % 100 + i);
  }

  // run prediction and get outputs
  std::vector<float> outputs = globalCache()->run(input_names_, data_, input_shapes_)[0];

  // print the input and output data
  std::cout << "input data -> ";
  for (auto &i: group_data) { std::cout << i << " "; }
  std::cout << std::endl << "output data -> ";
  for (auto &i: outputs) { std::cout << i << " "; }
  std::cout << std::endl;

}

DEFINE_FWK_MODULE(MyPlugin);

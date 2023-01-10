/*
 * Example plugin to demonstrate the direct single-threaded inference with TensorFlow 2.
 */

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

class MyPlugin : public edm::one::EDAnalyzer<> {
public:
  explicit MyPlugin(const edm::ParameterSet&);
  ~MyPlugin(){};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

private:
  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

  std::string graphPath_;
  std::string inputTensorName_;
  std::string outputTensorName_;

  tensorflow::GraphDef* graphDef_;
  tensorflow::Session* session_;
};

void MyPlugin::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // defining this function will lead to a *_cfi file being generated when compiling
  edm::ParameterSetDescription desc;
  desc.add<std::string>("graphPath");
  desc.add<std::string>("inputTensorName");
  desc.add<std::string>("outputTensorName");
  descriptions.addWithDefaultLabel(desc);
}

MyPlugin::MyPlugin(const edm::ParameterSet& config)
    : graphPath_(config.getParameter<std::string>("graphPath")),
      inputTensorName_(config.getParameter<std::string>("inputTensorName")),
      outputTensorName_(config.getParameter<std::string>("outputTensorName")),
      graphDef_(nullptr),
      session_(nullptr) {
  // set tensorflow log leven to warning
  tensorflow::setLogging("2");
}

void MyPlugin::beginJob() {
  // load the graph
  graphDef_ = tensorflow::loadGraphDef(graphPath_);

  // create a new session and add the graphDef
  session_ = tensorflow::createSession(graphDef_);
}

void MyPlugin::endJob() {
  // close the session
  tensorflow::closeSession(session_);

  // delete the graph
  delete graphDef_;
  graphDef_ = nullptr;
}

void MyPlugin::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // define a tensor and fill it with range(10)
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 10});
  for (size_t i = 0; i < 10; i++) {
    input.matrix<float>()(0, i) = float(i);
  }

  // define the output and run
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::run(session_, {{inputTensorName_, input}}, {outputTensorName_}, &outputs);

  // print the output
  std::cout << " -> " << outputs[0].matrix<float>()(0, 0) << std::endl << std::endl;
}

DEFINE_FWK_MODULE(MyPlugin);

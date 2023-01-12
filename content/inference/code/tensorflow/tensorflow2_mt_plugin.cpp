/*
 * Example plugin to demonstrate the direct multi-threaded inference with TensorFlow 2.
 */

#include <memory>

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/MakerMacros.h"
#include "FWCore/Framework/interface/stream/EDAnalyzer.h"
#include "FWCore/ParameterSet/interface/ParameterSet.h"
#include "PhysicsTools/TensorFlow/interface/TensorFlow.h"

// put a tensorflow::SessionCache into the global cache structure
// the session cache wraps both a tf graph and a tf session instance and also handles their deletion
class MyPlugin : public edm::stream::EDAnalyzer<edm::GlobalCache<tensorflow::SessionCache>> {
public:
  explicit MyPlugin(const edm::ParameterSet&, const tensorflow::SessionCache*);
  ~MyPlugin(){};

  static void fillDescriptions(edm::ConfigurationDescriptions&);

  // an additional static method for initializing the global cache
  static std::unique_ptr<tensorflow::SessionCache> initializeGlobalCache(const edm::ParameterSet&);

private:
  void beginJob();
  void analyze(const edm::Event&, const edm::EventSetup&);
  void endJob();

  std::string inputTensorName_;
  std::string outputTensorName_;

  // a pointer to the session created by the global session cache
  const tensorflow::Session* session_;
};

std::unique_ptr<tensorflow::SessionCache> MyPlugin::initializeGlobalCache(const edm::ParameterSet& config) {
  // this method is supposed to create, initialize and return a SessionCache instance
  std::string graphPath = edm::FileInPath(params.getParameter<std::string>("graphPath")).fullPath();
  return std::make_unique<tensorflow::SessionCache>(graphPath);
}

void MyPlugin::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  // defining this function will lead to a *_cfi file being generated when compiling
  edm::ParameterSetDescription desc;
  desc.add<std::string>("graphPath");
  desc.add<std::string>("inputTensorName");
  desc.add<std::string>("outputTensorName");
  descriptions.addWithDefaultLabel(desc);
}

MyPlugin::MyPlugin(const edm::ParameterSet& config,  const tensorflow::SessionCache* cache)
    : inputTensorName_(config.getParameter<std::string>("inputTensorName")),
      outputTensorName_(config.getParameter<std::string>("outputTensorName")),
      session_(cache->getSession()) {}

void MyPlugin::beginJob() {}

void MyPlugin::endJob() {
  // close the session
  tensorflow::closeSession(session_);
}

void MyPlugin::analyze(const edm::Event& event, const edm::EventSetup& setup) {
  // define a tensor and fill it with range(10)
  tensorflow::Tensor input(tensorflow::DT_FLOAT, {1, 10});
  for (size_t i = 0; i < 10; i++) {
    input.matrix<float>()(0, i) = float(i);
  }

  // define the output
  std::vector<tensorflow::Tensor> outputs;

  // evaluate
  // note: in case this line causes the compile to complain about the const'ness of the session_ in
  //       this call, your CMSSW version might not yet support passing a const session, so in this
  //       case, pass "const_cast<tensorflow::Session*>(session_)"
  tensorflow::run(session_, {{inputTensorName_, input}}, {outputTensorName_}, &outputs);

  // print the output
  std::cout << " -> " << outputs[0].matrix<float>()(0, 0) << std::endl << std::endl;
}

DEFINE_FWK_MODULE(MyPlugin);

from PhysicsTools.NanoAODTools.postprocessing.framework.datamodel import Collection
from PhysicsTools.NanoAODTools.postprocessing.framework.eventloop import Module
import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True

import onnxruntime as ort
import numpy as np
import os 

class exampleOrtProducer(Module):
    def __init__(self):
        pass

    def beginJob(self):
        model_path = os.path.join(os.getenv("CMSSW_BASE"), 'src', 'MySubsystem/MyModule/data/model.onnx')
        self.ort_sess = ort.InferenceSession(model_path)

    def endJob(self):
        pass

    def beginFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        self.out = wrappedOutputTree
        self.out.branch("OrtScore", "F")

    def endFile(self, inputFile, outputFile, inputTree, wrappedOutputTree):
        pass

    def analyze(self, event):
        """process event, return True (go to next module) or False (fail, go to next event)"""

        # create input data
        data = np.arange(event.event % 100, event.event % 100 + 10).astype(np.float32)
        # run inference
        outputs = self.ort_sess.run(None, {'my_input': np.array([data])})[0]
        # print input and output
        print('input ->', data)
        print('output ->', outputs)

        self.out.fillBranch("OrtScore", outputs[0][0])
        return True


# define modules using the syntax 'name = lambda : constructor' to avoid having them loaded when not needed

exampleOrtModuleConstr = lambda: exampleOrtProducer()

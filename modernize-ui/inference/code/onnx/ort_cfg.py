# coding: utf-8

import os

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


# setup minimal options
options = VarParsing("python")
options.setDefault("inputFiles", "/store/mc/RunIISummer20UL18MiniAODv2/DYJetsToLL_M-50_TuneCP5_13TeV-amcatnloFXFX-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/230000/4C8619B2-D0C0-4647-B946-B33754F4ED16.root")  # noqa
options.parseArguments()

# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
process.source = cms.Source("PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles))

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(True),
)

# setup options for multithreaded
process.options.numberOfThreads=cms.untracked.uint32(1)
process.options.numberOfStreams=cms.untracked.uint32(0)
process.options.numberOfConcurrentLuminosityBlocks=cms.untracked.uint32(1)


# setup MyPlugin by loading the auto-generated cfi (see MyPlugin.fillDescriptions)
process.load("MySubsystem.MyModule.myPlugin_cfi")
# specify the path of the ONNX model
process.myPlugin.model_path = "MySubsystem/MyModule/data/model.onnx"
# input names as defined in the model
# the order of name strings should also corresponds to the order of input data array feed to the model
process.myPlugin.input_names = ["my_input"]

# define what to run in the path
process.p = cms.Path(process.myPlugin)

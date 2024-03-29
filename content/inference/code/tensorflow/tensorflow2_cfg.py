# coding: utf-8

import os

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


# get the data/ directory
thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "data")

# setup minimal options
options = VarParsing("python")
options.setDefault("inputFiles", "root://xrootd-cms.infn.it//store/mc/RunIISummer20UL17MiniAODv2/TTToSemiLeptonic_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_mc2017_realistic_v9-v1/00000/005708B7-331C-904E-88B9-189011E6C9DD.root")  # noqa
options.parseArguments()

# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(
    input=cms.untracked.int32(10),
)
process.source = cms.Source(
    "PoolSource",
    fileNames=cms.untracked.vstring(options.inputFiles),
)

# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(True),
)

# setup MyPlugin by loading the auto-generated cfi (see MyPlugin.fillDescriptions)
process.load("MySubsystem.MyModule.myPlugin_cfi")
process.myPlugin.graphPath = cms.string(os.path.join(datadir, "graph.pb"))
process.myPlugin.inputTensorName = cms.string("input")
process.myPlugin.outputTensorName = cms.string("output")

# define what to run in the path
process.p = cms.Path(process.myPlugin)

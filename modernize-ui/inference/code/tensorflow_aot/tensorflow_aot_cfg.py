# coding: utf-8

import os

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing


# get the data/ directory
thisdir = os.path.dirname(os.path.abspath(__file__))
datadir = os.path.join(os.path.dirname(thisdir), "data")

# setup minimal options
options = VarParsing("python")
options.setDefault("inputFiles", "root://xrootd-cms.infn.it//store/mc/RunIISummer20UL18MiniAODv2/GluGluToHHTo2G2Tau_node_cHHH1_TuneCP5_13TeV-powheg-pythia8/MINIAODSIM/106X_upgrade2018_realistic_v16_L1v1-v2/40000/871B714B-0AA3-3342-B56C-6DC0E634593A.root")  # noqa
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
# register three batch rules
# - add 1+1+1 for batch size 3
# - add 4+1 for batch size 5
# - add 4+4 for batch size 6, using a padding of 2 for the second call
process.myPlugin.batchRules = cms.vstring(["3:1,1,1", "5:4,1", "6:4,4"])

# define what to run in the path
process.p = cms.Path(process.myPlugin)

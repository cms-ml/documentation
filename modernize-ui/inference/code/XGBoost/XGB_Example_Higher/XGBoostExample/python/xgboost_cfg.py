# coding: utf-8

import os

import FWCore.ParameterSet.Config as cms
from FWCore.ParameterSet.VarParsing import VarParsing

# setup minimal options
#options = VarParsing("python")
#options.setDefault("inputFiles", "root://xrootd-cms.infn.it//store/mc/RunIIFall17MiniAOD/DYJetsToLL_M-10to50_TuneCP5_13TeV-madgraphMLM-pythia8/MINIAODSIM/94X_mc2017_realistic_v10-v2/00000/9A439935-1FFF-E711-AE07-D4AE5269F5FF.root")  # noqa
#options.parseArguments()

# define the process to run
process = cms.Process("TEST")

# minimal configuration
process.load("FWCore.MessageService.MessageLogger_cfi")
process.MessageLogger.cerr.FwkReport.reportEvery = 1
process.maxEvents = cms.untracked.PSet(input=cms.untracked.int32(10))
#process.source = cms.Source("PoolSource",
#    fileNames=cms.untracked.vstring('file:/afs/cern.ch/cms/Tutorials/TWIKI_DATA/TTJets_8TeV_53X.root'))
process.source = cms.Source("EmptySource")
#process.source = cms.Source("PoolSource",
#    fileNames=cms.untracked.vstring(options.inputFiles))
# process options
process.options = cms.untracked.PSet(
    allowUnscheduled=cms.untracked.bool(True),
    wantSummary=cms.untracked.bool(True),
)

process.XGBoostExample = cms.EDAnalyzer("XGBoostExample")

# setup MyPlugin by loading the auto-generated cfi (see MyPlugin.fillDescriptions)
#process.load("XGB_Example.XGBoostExample.XGBoostExample_cfi")
process.XGBoostExample.model_path = cms.string("/Your/Path/data/highVer.model")  
process.XGBoostExample.test_data_path = cms.string("/Your/Path/data/Test_data.csv")

# define what to run in the path
process.p = cms.Path(process.XGBoostExample)

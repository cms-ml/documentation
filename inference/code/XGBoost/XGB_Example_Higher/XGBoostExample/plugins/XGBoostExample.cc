// -*- C++ -*-
//
// Package:    XGB_Example/XGBoostExample
// Class:      XGBoostExample
//
/**\class XGBoostExample XGBoostExample.cc XGB_Example/XGBoostExample/plugins/XGBoostExample.cc

 Description: [one line class summary]

 Implementation:
     [Notes on implementation]
*/
//
// Original Author:  Qian Sitian
//         Created:  Sat, 19 Jun 2021 08:38:51 GMT
//
//


// system include files
#include <memory>

// user include files
#include "FWCore/Framework/interface/Frameworkfwd.h"
#include "FWCore/Framework/interface/one/EDAnalyzer.h"

#include "FWCore/Framework/interface/Event.h"
#include "FWCore/Framework/interface/MakerMacros.h"

#include "FWCore/ParameterSet/interface/ParameterSet.h"
 #include "FWCore/Utilities/interface/InputTag.h"
 #include "DataFormats/TrackReco/interface/Track.h"
 #include "DataFormats/TrackReco/interface/TrackFwd.h"

#include <xgboost/c_api.h>
#include <vector>
#include <tuple>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>

using namespace std;

vector<vector<double>> readinCSV(const char* name){
	auto fin = ifstream(name);
	vector<vector<double>> floatVec;
	string strFloat;
	float fNum;
	int counter = 0;
	getline(fin,strFloat);
	while(getline(fin,strFloat))
	{
		std::stringstream  linestream(strFloat);
		floatVec.push_back(std::vector<double>());
		while(linestream>>fNum)
		{
			floatVec[counter].push_back(fNum);
			if (linestream.peek() == ',')
			linestream.ignore();
		}
		++counter;
	}
	return floatVec;
}

//
// class declaration
//

// If the analyzer does not use TFileService, please remove
// the template argument to the base class so the class inherits
// from  edm::one::EDAnalyzer<>
// This will improve performance in multithreaded jobs.



class XGBoostExample : public edm::one::EDAnalyzer<>  {
   public:
      explicit XGBoostExample(const edm::ParameterSet&);
      ~XGBoostExample();

      static void fillDescriptions(edm::ConfigurationDescriptions& descriptions);


   private:
      virtual void beginJob() ;
      virtual void analyze(const edm::Event&, const edm::EventSetup&) ;
      virtual void endJob() ;

      // ----------member data ---------------------------

	std::string test_data_path;
	std::string model_path;

	


};

//
// constants, enums and typedefs
//

//
// static data member definitions
//

//
// constructors and destructor
//
XGBoostExample::XGBoostExample(const edm::ParameterSet& config):
test_data_path(config.getParameter<std::string>("test_data_path")),
model_path(config.getParameter<std::string>("model_path"))
{

}


XGBoostExample::~XGBoostExample()
{

   // do anything here that needs to be done at desctruction time
   // (e.g. close files, deallocate resources etc.)

}


//
// member functions
//

void
XGBoostExample::analyze(const edm::Event& iEvent, const edm::EventSetup& iSetup)
{
}


void
XGBoostExample::beginJob()
{
	BoosterHandle booster_;
	XGBoosterCreate(NULL,0,&booster_);
	XGBoosterLoadModel(booster_,model_path.c_str());
	unsigned long numFeature = 0;
	vector<vector<double>> TestDataVector = readinCSV(test_data_path.c_str());
	float TestData[2000][8];
	for(unsigned i=0; (i < 2000); i++)
	{ 
		for(unsigned j=0; (j < 8); j++)
		{
			TestData[i][j] = TestDataVector[i][j];
		//	cout<<TestData[i][j]<<"\t";
		} 
		//cout<<endl;
	}
	DMatrixHandle data_;
	XGDMatrixCreateFromMat((float *)TestData,2000,8,-1,&data_);
	bst_ulong out_len=0;
	  const float *f;
	auto ret=XGBoosterPredict(booster_, data_,0, 0,0,&out_len,&f);
		  for (unsigned int i=0;i<out_len;i++)
			        std::cout <<  i << "\t"<< f[i] << std::endl;
}

void
XGBoostExample::endJob()
{
}

void
XGBoostExample::fillDescriptions(edm::ConfigurationDescriptions& descriptions) {
  //The following says we do not know what parameters are allowed so do no validation
  // Please change this to state exactly what you do use, even if it is no parameters
  edm::ParameterSetDescription desc;
  desc.add<std::string>("test_data_path");
  desc.add<std::string>("model_path");
  descriptions.addWithDefaultLabel(desc);

  //Specify that only 'tracks' is allowed
  //To use, remove the default given above and uncomment below
  //ParameterSetDescription desc;
  //desc.addUntracked<edm::InputTag>("tracks","ctfWithMaterialTracks");
  //descriptions.addDefault(desc);
}

//define this as a plug-in
DEFINE_FWK_MODULE(XGBoostExample);

#include "indri/ScoredExtentResult.hpp"
#include "indri/QueryEnvironment.hpp"
#include "indri/QueryParserFactory.hpp"
#include "indri/QueryAnnotation.hpp"
#include "indri/IndexEnvironment.hpp"
#include "indri/Parameters.hpp"
#include <math.h>
#include <string>
#include <ctype.h>
#include <ctime>
#include <sstream>
#include <queue>
#include <time.h>
#include "indri/LocalQueryServer.hpp"
#include "indri/delete_range.hpp"
#include "indri/NetworkStream.hpp"
#include "indri/NetworkMessageStream.hpp"
#include "indri/NetworkServerProxy.hpp"
#include "indri/ListIteratorNode.hpp"
#include "indri/ExtentInsideNode.hpp"
#include "indri/DocListIteratorNode.hpp"
#include "indri/FieldIteratorNode.hpp"
#include "indri/ParsedDocument.hpp"
#include "indri/Collection.hpp"
#include "indri/CompressedCollection.hpp"
#include "indri/TaggedDocumentIterator.hpp"
#include "indri/XMLNode.hpp"
#include "indri/QueryExpander.hpp"
#include "indri/RMExpander.hpp"
#include "indri/PonteExpander.hpp"
#include "indri/TFIDFExpander.hpp"
#include "indri/IndriTimer.hpp"
#include "indri/UtilityThread.hpp"
#include "indri/ScopedLock.hpp"
#include "indri/delete_range.hpp"
#include "indri/SnippetBuilder.hpp"
#include "indri/Porter_Stemmer.hpp"

using namespace lemur::api;
using namespace indri::api;

bool isNotAlpha(char c) 
{ 
	return !isalpha(c);
} 

/*
 * Code snippet copied from IndriRunQuery.cpp
 */

struct query_t {
	struct greater {
		bool operator() ( query_t* one, query_t* two ) {
			return one->index > two->index;
		}
	};

	query_t( int _index, std::string _number, const std::string& _text, const std::string &queryType,  std::vector<std::string> workSet,   std::vector<std::string> FBDocs) :
		index( _index ),
		number( _number ),
		text( _text ), qType(queryType), workingSet(workSet), relFBDocs(FBDocs)
	{
	}

	query_t( int _index, std::string _number, const std::string& _text ) :
		index( _index ),
		number( _number ),
		text( _text )
	{
	}

	std::string number;
	int index;
	std::string text;
	std::string qType;
	// working set to restrict retrieval
	std::vector<std::string> workingSet;
	// Rel fb docs
	std::vector<std::string> relFBDocs;
};


/*
 * code snippet copied as-is from IndriRunQuery.cpp
 */
void push_queue( std::queue< query_t* >& q, indri::api::Parameters& queries,int queryOffset ) {
std::cout<<"blah"<<std::endl;
	for( size_t i=0; i<queries.size(); i++ ) {
		std::string queryNumber;
		std::string queryText;
		std::string queryType = "indri";
		if( queries[i].exists( "type" ) )
			queryType = (std::string) queries[i]["type"];
		if (queries[i].exists("text"))
			queryText = (std::string) queries[i]["text"];
		if( queries[i].exists( "number" ) ) {
			queryNumber = (std::string) queries[i]["number"];
		} else {
			int thisQuery=queryOffset + int(i);
			std::stringstream s;
			s << thisQuery;
			queryNumber = s.str();
		}
		if (queryText.size() == 0)
			queryText = (std::string) queries[i];

		// working set and RELFB docs go here.
		// working set to restrict retrieval
		// UNUSED
		std::vector<std::string> workingSet;
		std::vector<std::string> relFBDocs;

		q.push( new query_t( i, queryNumber, queryText, queryType, workingSet, relFBDocs ) );

	}
}



/**
 * Main method, entry point of the application
 */
int main(int argc, char * argv[])
{
	int lnc2Instances = 0;
	int lnc2Correct = 0;

	if(argc<2){
		std::cerr<<"[ERROR] At least one parameter file needs to be provided"<<std::endl;
		return 0;
	}

	int split = atoi(argv[2]);

	std::cout<<"Parsing parameter file "<<argv[1]<<" ..."<<std::endl;
	indri::api::Parameters& param = indri::api::Parameters::instance();
	param.loadFile(argv[1]);

	//open the index (parameter <index>)
	//provides easy access to the TermList per document
	indri::collection::Repository repository;
	indri::index::Index *thisIndex;

	//provides easy access to compute retrieval status values
	indri::api::QueryEnvironment *queryEnvDIR = new indri::api::QueryEnvironment();
	indri::api::IndexEnvironment *indexEnv = new indri::api::IndexEnvironment();

	try
	{
		std::cout<<"Opening index at "<<param.get("index","")<<std::endl;
		repository.openRead(param.get("index",""));
		indri::collection::Repository::index_state repIndexState = repository.indexes();
    		thisIndex=(*repIndexState)[0];

		indexEnv->open(param.get("index",""));

		queryEnvDIR->addIndex(*indexEnv);

		std::vector<std::string> rulesDIR;
		rulesDIR.push_back("method:dirichlet");
		rulesDIR.push_back("mu:2500");
		queryEnvDIR->setScoringRules(rulesDIR);
	}
	catch (Exception &ex)
	{
		std::cerr<<"[ERROR] Something went wrong when opening the IndexEnv and QueryEnv!"<<std::endl;
		std::cout<<ex.what()<<std::endl;
		return 0;
	}

	std::cout<<"parsing queries"<<std::endl;

	//read the queries from file
	std::queue< query_t* > queries;
    	try {
		indri::api::Parameters parameterQueries = param[ "query" ];
		int queryOffset = param.get( "queryOffset", 0 );
		push_queue( queries, parameterQueries, queryOffset );
    		int queryCount = (int)queries.size();
   	 	std::cout<<"Number of queries read: "<<queryCount<<std::endl;
	}
	catch(Exception &ex){
		std::cout<<"[ERROR] query parsing issue!"<<std::endl;
		std::cout<<ex.what()<<std::endl;
	}
	int counter = 0;


	//process the queries: lowercasing and Porter stemming (hardcoded)
	while(!queries.empty())
	{
		counter++;
		query_t* query = queries.front();

		if(counter%split != 0){
			queries.pop();
			continue;
		}

		//move from query string to query-term vector
		std:vector<std::string> queryTokens;
	
		std::string normalizedQueryText;	
		std::string buf;
		stringstream ss(query->text);
		while(ss>>buf){

			//Porter stemming
			char *word = (char*)buf.c_str();

			//does the token appear in the index?
			if(queryEnvDIR->documentCount(word)>0)
			{
				queryTokens.push_back(word);
				if(normalizedQueryText.length()>0){
					normalizedQueryText.append(" ");
				}
				normalizedQueryText.append(word);
			}
			else {
				std::cerr<<"[WARNING] term "<<word<<" was _not_ found in the index!"<<std::endl;
			}
		}
		std::cout<<"Parsed query: ["<<normalizedQueryText<<"]"<<std::endl;
		
		//remove from vector queue
		queries.pop();

		//walk over the ranked list of documents
		//every time we hit a document that was ranked for the query we generate features
		std::ifstream documents(param.get("rankedDocsFile","").c_str());
		if(!documents.is_open()){
			std::cerr<<"[ERROR] Unable to open the document result file for reading!"<<std::endl;
			return 0;
		}

		bool queryDocsFound = false;
		int rank;
		std::string qid, docid, dummy, score, dummy2;//we also need qid, dummy, docid, rel (defined earlier)
		while(documents >> qid >> dummy >> docid >> rank >> score >> dummy2)
		{
			//right query?
			if(query->number.compare(qid)!=0 && queryDocsFound==false)
				continue;
			//we assume a sorted result file (all docs scored for a query follow each other)
			else if(query->number.compare(qid)!=0 && queryDocsFound==true)
				break; //we already saw a bunch of documents for that query, break
			else
				queryDocsFound = true;

			//convert external document identifier to an internal one 
			std::vector<std::string> externalDocVec;
			externalDocVec.push_back(docid);
			//a vector containing a single docid
			std::vector<int> internalDocVec = queryEnvDIR->documentIDsFromMetadata("docno",externalDocVec);

			int doclen = thisIndex->documentLength(internalDocVec[0]);
			if(doclen > (512/2)){
				continue;
			}

			std::vector<indri::api::ScoredExtentResult> resVec;
			try {
				resVec = queryEnvDIR->runQuery("#combine("+query->text+")",internalDocVec,1);
			}
			catch(Exception &ex)
			{
				std::cerr<<"[ERROR] queryEnvDIR caused a problem!"<<std::endl;
				return 0;
			}
			double unduplicatedScore = resVec[0].score;
			

			int k = (int)512/doclen;
			//duplicate the document k times ...

			//compute the duplicate document
			std::stringstream duplicate;
			duplicate<<"<DOC><DOCNO>DUP1</DOCNO><TEXT>";
			const indri::index::TermList *termList=thisIndex->termList(internalDocVec[0]);
			if (termList)
			{
				indri::utility::greedy_vector<lemur::api::TERMID_T > terms = termList->terms();
				for(int i=0; i<terms.size(); i++)
				{
					std::string term = thisIndex->term( termList->terms()[i]);
					for(int j=0; j<k; j++)
						duplicate<<" "<<term;					
				}
			}
			delete termList;	
			duplicate<<"</TEXT></DOC>";

			//add the duplicate and score it
			std::vector<indri::parse::MetadataPair> metadata;
			int dupDocid = indexEnv->addString(duplicate.str(),"trectext",metadata);
			if(dupDocid == 0){
				std::cerr<<"[ERROR] duplicate document couldn't be added!"<<std::endl;
			}	

			internalDocVec.clear();
			internalDocVec.push_back(dupDocid);
			resVec.clear();
			try {
				resVec = queryEnvDIR->runQuery("#combine("+query->text+")",internalDocVec,1);
			}
			catch(Exception &ex)
			{
				std::cerr<<"[ERROR] queryEnvDIR caused a problem again!"<<std::endl;
				return 0;
			}
			double duplicatedScore = resVec[0].score;

			lnc2Instances++;
			if(unduplicatedScore<=duplicatedScore)
				lnc2Correct++;

			double fraction = (double)lnc2Correct/(double)lnc2Instances;
			std::cout<<"\t\t"<<unduplicatedScore<<" vs. "<<duplicatedScore<<std::endl;
		}
		documents.close();
		
		std::cout<<"Instances: "<<lnc2Instances<<", correct: "<<lnc2Correct<<std::endl;
	}

	return 0;
}


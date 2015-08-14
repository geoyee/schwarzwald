
#ifndef CLOUDJS_H
#define CLOUDJS_H


#include <string>
#include <vector>
#include <sstream>
#include <list>

#include "rapidjson/document.h"
#include "rapidjson/prettywriter.h"
#include "rapidjson/stringbuffer.h"

#include "AABB.h"
#include "definitions.hpp"
#include "PointAttributes.hpp"

using std::string;
using std::vector;
using std::stringstream;
using std::list;
using rapidjson::Document;
using rapidjson::StringBuffer;
using rapidjson::Writer;
using rapidjson::PrettyWriter;
using rapidjson::Value;

namespace Potree{

class CloudJS{
public:

	class Node{
	public:
		string name;
		int pointCount;

		Node(string name, int pointCount){
			this->name = name;
			this->pointCount = pointCount;
		}
	};

	string version;
	string octreeDir;
	AABB boundingBox;
	AABB tightBoundingBox;
	OutputFormat outputFormat;
	PointAttributes pointAttributes;
	double spacing;
	vector<Node> hierarchy;
	double scale;
	int hierarchyStepSize;

	CloudJS(){
		version = "0.0";
		hierarchyStepSize = -1;
	}

	CloudJS(string content){
		Document d;
		d.Parse(content.c_str());

		Value &vVersion = d["version"];
		Value &vOctreeDir = d["octreeDir"];
		Value &vBoundingBox = d["boundingBox"];
		Value &vTightBoundingBox = d["tightBoundingBox"];
		Value &vPointAttributes = d["pointAttributes"];
		Value &vSpacing = d["spacing"];
		Value &vScale = d["scale"];
		Value &vHierarchyStepSize = d["hierarchyStepSize"];

		version = vVersion.GetString();
		octreeDir = vOctreeDir.GetString();
		boundingBox = AABB(
			Vector3<double>(vBoundingBox["lx"].GetDouble(), vBoundingBox["ly"].GetDouble(), vBoundingBox["lz"].GetDouble()),
			Vector3<double>(vBoundingBox["lx"].GetDouble(), vBoundingBox["ly"].GetDouble(), vBoundingBox["lz"].GetDouble())
		);
		tightBoundingBox = AABB(
			Vector3<double>(vTightBoundingBox["lx"].GetDouble(), vTightBoundingBox["ly"].GetDouble(), vTightBoundingBox["lz"].GetDouble()),
			Vector3<double>(vTightBoundingBox["lx"].GetDouble(), vTightBoundingBox["ly"].GetDouble(), vTightBoundingBox["lz"].GetDouble())
		);

		if(vPointAttributes.IsArray()){
			outputFormat = OutputFormat::BINARY;
			pointAttributes = PointAttributes();

			for (Value::ConstValueIterator itr = vPointAttributes.Begin(); itr != vPointAttributes.End(); ++itr){
				string strpa = itr->GetString();
				PointAttribute pa = PointAttribute::fromString(strpa);
				pointAttributes.add(pa);
			}


		}else{
			string pa = vPointAttributes.GetString();
			if(pa == "LAS"){
				outputFormat = OutputFormat::LAS;
			}else if(pa == "LAZ"){
				outputFormat = OutputFormat::LAZ;
			}
		}

		spacing = vSpacing.GetDouble();
		scale = vScale.GetDouble();
		hierarchyStepSize = vHierarchyStepSize.GetInt();

	}

	string getString(){

		Document d(rapidjson::kObjectType);

		Value version(this->version.c_str(), this->version.size());
		Value octreeDir("data");
		Value boundingBox(rapidjson::kObjectType);
		{
			//Value min(rapidjson::kArrayType);
			//min.PushBack(this->boundingBox.min.x, d.GetAllocator());
			//min.PushBack(this->boundingBox.min.y, d.GetAllocator());
			//min.PushBack(this->boundingBox.min.z, d.GetAllocator());
			//
			//Value max(rapidjson::kArrayType);
			//max.PushBack(this->boundingBox.max.x, d.GetAllocator());
			//max.PushBack(this->boundingBox.max.y, d.GetAllocator());
			//max.PushBack(this->boundingBox.max.z, d.GetAllocator());
			//
			//boundingBox.AddMember("min", min, d.GetAllocator());
			//boundingBox.AddMember("max", max, d.GetAllocator());

			boundingBox.AddMember("lx", this->boundingBox.min.x, d.GetAllocator());
			boundingBox.AddMember("ly", this->boundingBox.min.y, d.GetAllocator());
			boundingBox.AddMember("lz", this->boundingBox.min.z, d.GetAllocator());
			boundingBox.AddMember("ux", this->boundingBox.max.x, d.GetAllocator());
			boundingBox.AddMember("uy", this->boundingBox.max.y, d.GetAllocator());
			boundingBox.AddMember("uz", this->boundingBox.max.z, d.GetAllocator());
		}
		Value tightBoundingBox(rapidjson::kObjectType);
		{
			//Value min(rapidjson::kArrayType);
			//min.PushBack(this->tightBoundingBox.min.x, d.GetAllocator());
			//min.PushBack(this->tightBoundingBox.min.y, d.GetAllocator());
			//min.PushBack(this->tightBoundingBox.min.z, d.GetAllocator());
			//
			//Value max(rapidjson::kArrayType);
			//max.PushBack(this->tightBoundingBox.max.x, d.GetAllocator());
			//max.PushBack(this->tightBoundingBox.max.y, d.GetAllocator());
			//max.PushBack(this->tightBoundingBox.max.z, d.GetAllocator());
			//
			//tightBoundingBox.AddMember("min", min, d.GetAllocator());
			//tightBoundingBox.AddMember("max", max, d.GetAllocator());

			tightBoundingBox.AddMember("lx", this->tightBoundingBox.min.x, d.GetAllocator());
			tightBoundingBox.AddMember("ly", this->tightBoundingBox.min.y, d.GetAllocator());
			tightBoundingBox.AddMember("lz", this->tightBoundingBox.min.z, d.GetAllocator());
			tightBoundingBox.AddMember("ux", this->tightBoundingBox.max.x, d.GetAllocator());
			tightBoundingBox.AddMember("uy", this->tightBoundingBox.max.y, d.GetAllocator());
			tightBoundingBox.AddMember("uz", this->tightBoundingBox.max.z, d.GetAllocator());
		}
		Value pointAttributes;
		if(outputFormat == OutputFormat::BINARY){
			pointAttributes.SetArray();
			for(int i = 0; i < this->pointAttributes.size(); i++){
				PointAttribute attribute = this->pointAttributes[i];
				Value str(attribute.name.c_str(), d.GetAllocator());
				pointAttributes.PushBack(str, d.GetAllocator());
			}
		}else if(outputFormat == OutputFormat::LAS){
			pointAttributes = "LAS";
		}else if(outputFormat == OutputFormat::LAZ){
			pointAttributes = "LAZ";
		}
		Value spacing(this->spacing);
		Value scale(this->scale);
		Value hierarchyStepSize(this->hierarchyStepSize);


		d.AddMember("version", version, d.GetAllocator());
		d.AddMember("octreeDir", octreeDir, d.GetAllocator());
		d.AddMember("boundingBox", boundingBox, d.GetAllocator());
		d.AddMember("tightBoundingBox", tightBoundingBox, d.GetAllocator());
		d.AddMember("pointAttributes", pointAttributes, d.GetAllocator());
		d.AddMember("spacing", spacing, d.GetAllocator());
		d.AddMember("scale", scale, d.GetAllocator());
		d.AddMember("hierarchyStepSize", hierarchyStepSize, d.GetAllocator());

		StringBuffer buffer;
		PrettyWriter<StringBuffer> writer(buffer);
		d.Accept(writer);

		return buffer.GetString();
		



		//stringstream cloudJs;

		//cloudJs.precision(15);
		//cloudJs << "{" << endl;
		//cloudJs << "\t" << "\"version\": \"" << version << "\"," << endl;
		//cloudJs << "\t" << "\"octreeDir\": \"data\"," << endl;
		//cloudJs << "\t" << "\"boundingBox\": {" << endl;
		//cloudJs << "\t\t" << "\"lx\": " << boundingBox.min.x << "," << endl;
		//cloudJs << "\t\t" << "\"ly\": " << boundingBox.min.y << "," << endl;
		//cloudJs << "\t\t" << "\"lz\": " << boundingBox.min.z << "," << endl;
		//cloudJs << "\t\t" << "\"ux\": " << boundingBox.max.x << "," << endl;
		//cloudJs << "\t\t" << "\"uy\": " << boundingBox.max.y << "," << endl;
		//cloudJs << "\t\t" << "\"uz\": " << boundingBox.max.z << endl;
		//cloudJs << "\t" << "}," << endl;
		//cloudJs << "\t" << "\"tightBoundingBox\": {" << endl;
		//cloudJs << "\t\t" << "\"lx\": " << tightBoundingBox.min.x << "," << endl;
		//cloudJs << "\t\t" << "\"ly\": " << tightBoundingBox.min.y << "," << endl;
		//cloudJs << "\t\t" << "\"lz\": " << tightBoundingBox.min.z << "," << endl;
		//cloudJs << "\t\t" << "\"ux\": " << tightBoundingBox.max.x << "," << endl;
		//cloudJs << "\t\t" << "\"uy\": " << tightBoundingBox.max.y << "," << endl;
		//cloudJs << "\t\t" << "\"uz\": " << tightBoundingBox.max.z << endl;
		//cloudJs << "\t" << "}," << endl;
		//if(outputFormat == OutputFormat::BINARY){
		//	cloudJs << "\t" << "\"pointAttributes\": [" << endl;

		//	for(int i = 0; i < pointAttributes.size(); i++){
		//		PointAttribute attribute = pointAttributes[i];

		//		cloudJs << "\t\t" << "\"" << attribute.name << "\"";

		//		if(i+1 < pointAttributes.size()){
		//			cloudJs << ",";
		//		}
		//		cloudJs << endl;
		//	}

		//	cloudJs << "\t" << "]," << endl;
		//}else if(outputFormat == OutputFormat::LAS){
		//	cloudJs << "\t" << "\"pointAttributes\": \"LAS\"," << endl;
		//}else if(outputFormat == OutputFormat::LAZ){
		//	cloudJs << "\t" << "\"pointAttributes\": \"LAZ\"," << endl;
		//}
		//cloudJs << "\t" << "\"spacing\": " << spacing << "," << endl;
		//cloudJs << "\t" << "\"scale\": " << scale << "," << endl;
		//if(hierarchyStepSize >= 0){
		//	cloudJs << "\t" << "\"hierarchyStepSize\": " << hierarchyStepSize << endl;
		//}
		////cloudJs << "\t" << "\"hierarchy\": [" << endl;
		////
		////for(int i = 0; i < hierarchy.size(); i++){
		////	Node node = hierarchy[i];
		////	cloudJs << "\t\t[\"" << node.name << "\", " << node.pointCount << "]";
		////
		////	if(i < hierarchy.size()-1){
		////		cloudJs << ",";
		////	}
		////	cloudJs << endl;
		////}
		////cloudJs << "\t]" << endl;
		//cloudJs << "}" << endl;

		//return cloudJs.str();
	}
};

}

#endif

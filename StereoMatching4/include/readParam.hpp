#pragma once
#include <iostream>
#include <fstream>
#include <string>
#include <map>

class ParameterReader
{
public:
	ParameterReader(std::string filename="./parameters.txt")
	{
		std::ifstream fin(filename.c_str());
		if(!fin)
		{
			std::cerr<<"parameter file does not exist."<<std::endl;
			return;
		}
		while(!fin.eof())
		{
			std::string str;
			getline(fin,str);
			if(str[0]=='#') //以#开头的是注释
			{
				continue;
			}
			int pos=str.find("=");
			if(pos==-1)
				continue;
			std::string key=str.substr(0,pos);
			std::string value = str.substr(pos+1,str.length());
			data[key] = value;

			if(!fin.good())
				break;
		}
	}
	std::string getData(std::string key)
	{
		std::map<std::string,std::string>::iterator iter=data.find(key);
		if(iter==data.end())
		{
			std::cerr<<"Parameter name "<<key<<" not found!"<<std::endl;
			return std::string("NOT FOUND");
		}
		return iter->second;
	}
public:
	std::map<std::string,std::string> data;
};

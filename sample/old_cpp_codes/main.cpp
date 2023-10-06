/*
2022 - December
Mini-Unity --------- 基於Unity3D運行模式的模擬引擎。

SID:1155159003
Name:Chan Tai Ming
Email: yashin.sd123@yahoo.com.hk / 1155159003@link.cuhk.edu
Phone: +852 6502 6772 / +86 13143884863

*/
#pragma once
#include "Engine.h"
#include <iostream>
using namespace std;

int main(int argc, char* argv[])
{	
	//Engine::TestEngine();
		
	Engine::SetGameName("mini unity test 1");
	if (Engine::StartEngine() < 0) {
		cout << "Error occur during StartEngine()" << endl;
		return -1;
	}
	if (Engine::BeforeEngineRun() < 0) {
		cout << "Error occur during BeforeEngineRun()" << endl;
		return -1;
	}
	if (Engine::RunEngine() < 0) {
		cout << "Error occur during RunEngine()" << endl;
		return -1;
	}
	Engine::EndEngine();

	return 0;
}







#pragma once
#include "GameCore.hpp"
#include "GameResource.hpp"
#include "Engine.h"

#pragma region RendererBasic

class RendererBasic {
protected:
	virtual void SubmitTaskToEngine()=0;
};

#pragma endregion




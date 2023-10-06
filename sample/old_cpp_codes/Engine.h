#pragma once
#include "Dependencies/glew/glew.h"
#include "Dependencies/GLFW/glfw3.h"
#include "DataStructure.hpp"
#include <concurrent_queue.h>
#include <chrono>
#include <ratio>
#include <thread>

//給Engine及InputHandler繼承的類，保護監聽器
class EngineEvents {
protected:
	static Event<keyCallbackData> KeyCallback;
	static Event<mouseCallbackData> MouseCallback;
	static Event<scrollCallbackData> ScrollCallback;
	static Event<cursorPositionCallbackData> CursorPositionCallback;
};

class GameObject;
/// <summary>
/// 引擎框架
/// </summary>
class Engine: public EngineEvents{
private:
	//setting parameter
	static unique_list<GameObject*>* RootObjects;
	static int scr_width; //default screen width
	static int scr_height; //default screen height
	static string gameName;
	static unsigned int lightWithDepthMap_maxNum; //max light number , which using depth map in game
	static unsigned int light_maxNum; //max light number in game
	static unsigned int Ortho_Distance;
	static unsigned int MaxFrameRate;
	static unsigned int MaxFixedUpdatePerSecond;
	static unsigned int shadowMapTextureWidth;
	static unsigned int shadowMapTextureHeight;
	static int currentTextureBindingPosition;
	static Event<void> LoadGameEvent;
	
	//private field
	static EngineStatus status;
	static GLFWwindow* window;
	static GLuint quadVAO_forFinalOutput;
	static GLuint quadVBO_forFinalOutput;
	static chrono::duration<double, ratio<1, 1000>> deltaTime;
	static chrono::duration<double, ratio<1, 1000>> timeCount_forFixedUpdate;
	static RenderFrameData tempRenderFrameData; //render frame data preparing in logic frame
	static GLuint texture2dArray_forDirectionalLight;
	static GLuint texture2dArray_forSpotLight;
	static GLuint textureCubeArray_forPointLight;
	static GLuint framebuffer_postRender;
	static GLuint textureColorbuffer_postRender;
	static GLuint depthStencilBuffer_postRender;
	
	static void framebuffer_size_callback(GLFWwindow* window, int width, int height);
	static void mouse_button_callback(GLFWwindow* window, int button, int action, int mods);
	static void cursor_position_callback(GLFWwindow* window, double x, double y);
	static void scroll_callback(GLFWwindow* window, double xoffset, double yoffset);
	static void key_callback(GLFWwindow* window, int key, int scancode, int action, int mods);
	
	static void CreateNewRenderFrameData();
public:

	//set engine 
	static void SetScreenSize(int width, int height);
	static void SetMaxLightNum(unsigned int maxNum);
	static void SetOrthoDistance(unsigned int distance);
	static void SetMaxFrameRate(unsigned int maxFrameRate);
	static void SetMaxFixedUpdatePerSecond(unsigned int maxFixedUpdatePerSecond);
	static void SetGameName(string name);
	static void SetMaxLightNumWithDepthMap(unsigned int maxNum);
	static void SetShadowMapTextureSize(unsigned int width, unsigned int height);
	static void AddLoadGameEvent(function<void()> func);

	//get engine data
	static float GetAspectRatio();
	static unsigned int GetOrthoDistance();
	static int GetWindowWidth();
	static int GetWindowHeight();
	static GLFWwindow* GetWindow();
	static unsigned int GetMaxLightNumWithDepthMap();
	static unsigned int GetMaxLightNum();
	static unsigned int GetMaxFrameRate();
	static unsigned int GetMaxFixedUpdatePerSecond();
	static EngineStatus GetStatus();
	static double GetDeltaTime();
	static int GetFrameRate();
	static vec2 GetShadowMapTextureSize();
	static int GetCurrentTextureBindingPosition();

	/// <summary>
	/// Add Render Task to current render data. Order is only needed for Transparent render task.
	/// </summary>
	static void AddBindedRenderFuncToRenderFrameData(function<void()> bindedFunc, GameShader* shader,RenderType renderType, int order=0);
	static void AddCastingShadowMeshDataToRenderFrameData(currentMeshData meshData, GameTexture* diffuseTex);
	static void SetAmbientLightDataToRenderFrameData(Color ambientLight, float ambientPower);
	static void AddSingleLightDataToRenderFrameData(SingleLightData lightData);
	static void AddShadowLightDataToRenderFrameData(ShadowLightData_PointLight shadowLight);
	static void AddShadowLightDataToRenderFrameData(ShadowLightData_OtherLight shadowLight);
	static void SetCameraDataToRenderFrameData(CameraVPData data, Color camBackgruondColor);

	//for testing
	static void TestEngine();
	
	//start engine
	static int StartEngine();
	
	//before engine run
	static void LoadGame();
	static void printAllSettings();
	static void printOpenGLInfo();
	static int BeforeEngineRun();
	
	//run engine
	static void RenderFinalQuad();
	static int RunEngine();
	
	//end engine
	static void EndEngine();
};

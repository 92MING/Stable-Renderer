#pragma once
#include "InputHandler.hpp"
#include "GameCore.hpp"
#include "GameResource.hpp"
#include "Camera.hpp"
#include "Light.hpp"
#include "MeshRenderer.hpp"
using namespace std;

const Color Color::Black = Color(0.0f, 0.0f, 0.0f, 1.0f);
const Color Color::White = Color(1.0f, 1.0f, 1.0f, 1.0f);
const Color Color::Red = Color(1.0f, 0.0f, 0.0f, 1.0f);
const Color Color::Green = Color(0.0f, 1.0f, 0.0f, 1.0f);
const Color Color::Blue = Color(0.0f, 0.0f, 1.0f, 1.0f);
const Color Color::Yellow = Color(1.0f, 1.0f, 0.0f, 1.0f);
const Color Color::Orange = Color(1.0f, 0.5f, 0.0f, 1.0f);
const Color Color::Purple = Color(0.5f, 0.0f, 1.0f, 1.0f);
const Color Color::Cyan = Color(0.0f, 1.0f, 1.0f, 1.0f);
const Color Color::Magenta = Color(1.0f, 0.0f, 1.0f, 1.0f);
const Color Color::Gray = Color(0.5f, 0.5f, 0.5f, 1.0f);
const Color Color::Clear = Color(0.0f, 0.0f, 0.0f, 0.0f);

//engine events
Event<keyCallbackData> EngineEvents::KeyCallback;
Event<mouseCallbackData> EngineEvents::MouseCallback;
Event<scrollCallbackData> EngineEvents::ScrollCallback;
Event<cursorPositionCallbackData> EngineEvents::CursorPositionCallback;

//settings
unique_list<GameObject*>* Engine::RootObjects = nullptr;
int Engine::scr_width = 1080;
int Engine::scr_height = 720;
string Engine::gameName = "";
unsigned int Engine::lightWithDepthMap_maxNum = 8;
unsigned int Engine::light_maxNum = 64;
unsigned int Engine::Ortho_Distance = 30;
unsigned int Engine::MaxFrameRate = 120;
unsigned int Engine::MaxFixedUpdatePerSecond = 60;
unsigned int Engine::shadowMapTextureWidth = 2048;
unsigned int Engine::shadowMapTextureHeight = 2048;
Event<void> Engine::LoadGameEvent;

//private field
EngineStatus Engine::status = BEFORE_INIT;
GLFWwindow* Engine::window = nullptr;
chrono::duration<double, ratio<1, 1000>> Engine::deltaTime;
chrono::duration<double, ratio<1, 1000>> Engine::timeCount_forFixedUpdate = chrono::duration<double, ratio<1, 1000>>(0);
RenderFrameData Engine::tempRenderFrameData;
int Engine::currentTextureBindingPosition = 0;
GLuint Engine::quadVAO_forFinalOutput = NULL;
GLuint Engine::quadVBO_forFinalOutput = NULL;
GLuint Engine::texture2dArray_forDirectionalLight;
GLuint Engine::texture2dArray_forSpotLight = NULL;
GLuint Engine::textureCubeArray_forPointLight = NULL;
GLuint Engine::framebuffer_postRender = NULL;
GLuint Engine::textureColorbuffer_postRender = NULL;
GLuint Engine::depthStencilBuffer_postRender = NULL;
const mat4 shadowMap_biasMatrix = {
	  0.5, 0.0, 0.0, 0.0,
	  0.0, 0.5, 0.0, 0.0,
	  0.0, 0.0, 0.5, 0.0,
	  0.5, 0.5, 0.5, 1.0 };

//input callback
void Engine::framebuffer_size_callback(GLFWwindow* window, int width, int height) {
	glViewport(0, 0, width, height);
}
void Engine::mouse_button_callback(GLFWwindow* window, int button, int action, int mods) {
	MouseCallback.Invoke({ button, action });
}
void Engine::cursor_position_callback(GLFWwindow* window, double x, double y) {
	CursorPositionCallback.Invoke({ x, y });
}
void Engine::scroll_callback(GLFWwindow* window, double xoffset, double yoffset) {
	ScrollCallback.Invoke({ xoffset, yoffset });
}
void Engine::key_callback(GLFWwindow* window, int key, int scancode, int action, int mods) {
	KeyCallback.Invoke({ key, action });
}

//setting
void Engine::SetScreenSize(int width, int height) {
	if (status == BEFORE_INIT) {
		scr_height = height;
		scr_width = width;
	}
	else {
		cout << "Screen Size must be called before Engine Init" << endl;
	}
}
void Engine::SetGameName(string name) {
	if (status == BEFORE_INIT) {
		gameName = name;
	}
	else {
		cout << "Game Name must be called before Engine Init" << endl;
	}
}
void Engine::SetMaxLightNumWithDepthMap(unsigned int maxNum) {
	if (status < RUNNING) {
		lightWithDepthMap_maxNum = maxNum;
	}
	else {
		cout << "Max DepthMap Light Num only avaliable to change called before Engine Run" << endl;
	}
}
void Engine::SetMaxLightNum(unsigned int maxNum) {
	if (status < RUNNING) {
		light_maxNum = maxNum;
	}
	else {
		cout << "Max Light Num only avaliable to change before Engine Run" << endl;
	}
}
void Engine::SetOrthoDistance(unsigned int distance) {
	if (distance>0) Ortho_Distance = distance;
	else {
		cout << "Ortho Distance must be greater than 0" << endl;
	}
}
void Engine::SetMaxFrameRate(unsigned int maxFrameRate) {
	MaxFrameRate = maxFrameRate;
}
void Engine::SetMaxFixedUpdatePerSecond(unsigned int maxFixedUpdatePerSecond) {
	MaxFixedUpdatePerSecond = maxFixedUpdatePerSecond;
}
void Engine::SetShadowMapTextureSize(unsigned int width, unsigned int height) {
	if (status < RUNNING) {
		shadowMapTextureWidth = width;
		shadowMapTextureHeight = height;
	}
	else {
		cout << "Shadow Map Texture Size only avaliable to change before Engine Run" << endl;
	}
}
void Engine::AddLoadGameEvent(function<void()> func) {
	LoadGameEvent.AddListener(func);
}

float Engine::GetAspectRatio() {
	return (float)scr_width / (float)scr_height;
}
unsigned int Engine::GetOrthoDistance() {
	return Ortho_Distance;
}
int Engine::GetWindowWidth() {
	return scr_width;
}
int Engine::GetWindowHeight() {
	return scr_height;
}
GLFWwindow* Engine::GetWindow() {
	return window;
}
unsigned int Engine::GetMaxLightNumWithDepthMap() {
	return lightWithDepthMap_maxNum;
}
unsigned int Engine::GetMaxLightNum() {
	return light_maxNum;
}
unsigned int Engine::GetMaxFrameRate() {
	return MaxFrameRate;
}
unsigned int Engine::GetMaxFixedUpdatePerSecond() {
	return MaxFixedUpdatePerSecond;
}
EngineStatus Engine::GetStatus() {
	return status;
}
double Engine::GetDeltaTime() {
	return deltaTime.count()/1000;
}
int Engine::GetFrameRate() {
	if (deltaTime.count() == 0) return MaxFrameRate;
	return (int)(1000 / deltaTime.count());
}
vec2 Engine::GetShadowMapTextureSize() {
	return vec2(shadowMapTextureWidth, shadowMapTextureHeight);
}
int Engine::GetCurrentTextureBindingPosition() {
	return currentTextureBindingPosition;
}

void Engine::CreateNewRenderFrameData() {
	tempRenderFrameData = RenderFrameData();
}
/**
void Engine::SubmitRenderFrameData() {
	
	RenderTaskQueue.push(tempRenderFrameData);
	//cout << "(logic loop)submitted frame data" << endl;
};
**/
void Engine::AddBindedRenderFuncToRenderFrameData(function<void()> bindedFunc, GameShader* shader, RenderType renderType, int order){
	if (renderType == Opaque) {
		tempRenderFrameData.opaqueRenderFuncs.push_back(pair<function<void()>,GameShader*>{bindedFunc, shader});
	}
	else if (renderType == Transparent) {
		tempRenderFrameData.transparentRenderFuncs.insert(pair<int, pair<function<void()>, GameShader*>>{order, pair<function<void()>, GameShader*>{bindedFunc, shader}});
	}
	else if (renderType == UI) {
		//從前面推入，讀取的時候可以直接從前面開始for...（越往後UI越優先，因為用深度緩衝可以省略後面被覆蓋的UI的渲染）
		tempRenderFrameData.uiRenderFuncs.push_front(pair<function<void()>, GameShader*>{bindedFunc, shader});
	}
};
void Engine::AddCastingShadowMeshDataToRenderFrameData(currentMeshData meshData, GameTexture* diffuseTex) {
	tempRenderFrameData.allMeshesCastingShadow.push_back(pair<currentMeshData, GameTexture*>{meshData, diffuseTex});
}
void Engine::SetAmbientLightDataToRenderFrameData(Color ambientLight, float ambientPower) {
	tempRenderFrameData.ambientLight = ambientLight;
	tempRenderFrameData.ambientPower = ambientPower;
}
void Engine::AddSingleLightDataToRenderFrameData(SingleLightData lightData) {
	tempRenderFrameData.allSingleLights.push_back(lightData);
};
void Engine::AddShadowLightDataToRenderFrameData(ShadowLightData_PointLight shadowLight) {
	tempRenderFrameData.allShadowLights_PointLight.push_back(shadowLight);
}
void Engine::AddShadowLightDataToRenderFrameData(ShadowLightData_OtherLight shadowLight) {
	if (shadowLight.lightType == DIRECTION_LIGHT) {
		tempRenderFrameData.allShadowLights_DirectionalLight.push_back(shadowLight);
	}
	else if (shadowLight.lightType == SPOT_LIGHT) {
		tempRenderFrameData.allShadowLights_SpotLight.push_back(shadowLight);
	}
}
void Engine::SetCameraDataToRenderFrameData(CameraVPData data, Color camBackgroundColor) {
	tempRenderFrameData.cameraVPData = data;
	tempRenderFrameData.cameraBackgroundColor = camBackgroundColor;
};
/**
RenderFrameData Engine::PopRenderTaskFromQueue() {
	RenderFrameData output;
	//wait for get a task
	while (!RenderTaskQueue.try_pop(output))
		this_thread::sleep_for(chrono::milliseconds(5));
	//get the latest task
	while (RenderTaskQueue.try_pop(output));
	return output;
}
**/

#pragma region test engine
void Engine::TestEngine() {
	if (!glfwInit()) {
		cout << "Failed to initialize GLFW" << endl;
		return;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
	GLFWwindow* window = glfwCreateWindow(1080, 720, "Test", NULL, NULL);
	if (!window) {
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return;
	}
	glfwMakeContextCurrent(window);
	if (GLEW_OK != glewInit()) {
		std::cout << "Failed to initialize GLEW" << std::endl;
		return;
	}
	//glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	const GLubyte* name = glGetString(GL_VENDOR);
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* glversion = glGetString(GL_VERSION);
	std::cout << "OpenGL company: " << name << std::endl;
	std::cout << "Renderer name: " << renderer << std::endl;
	std::cout << "OpenGL version: " << glversion << std::endl;
	
	GLuint cubeVAO = 0;
	GLuint cubeVBO = 0;
	GLfloat vertices[] = {
		// Back face
		-0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f, // Bottom-left
		0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f, // top-right
		0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 0.0f, // bottom-right         
		0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 1.0f, 1.0f,  // top-right
		-0.5f, -0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 0.0f,  // bottom-left
		-0.5f, 0.5f, -0.5f, 0.0f, 0.0f, -1.0f, 0.0f, 1.0f,// top-left
		// Front face
		-0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f, // bottom-left
		0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 0.0f,  // bottom-right
		0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f,  // top-right
		0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, // top-right
		-0.5f, 0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 1.0f,  // top-left
		-0.5f, -0.5f, 0.5f, 0.0f, 0.0f, 1.0f, 0.0f, 0.0f,  // bottom-left
		// Left face
		-0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // top-right
		-0.5f, 0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top-left
		-0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // bottom-left
		-0.5f, -0.5f, -0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 1.0f, // bottom-left
		-0.5f, -0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 0.0f, 0.0f,  // bottom-right
		-0.5f, 0.5f, 0.5f, -1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // top-right
		// Right face
		0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f, // top-left
		0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f, // bottom-right
		0.5f, 0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 1.0f, // top-right         
		0.5f, -0.5f, -0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 1.0f,  // bottom-right
		0.5f, 0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 1.0f, 0.0f,  // top-left
		0.5f, -0.5f, 0.5f, 1.0f, 0.0f, 0.0f, 0.0f, 0.0f, // bottom-left     
		// Bottom face
		-0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f, // top-right
		0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 1.0f, // top-left
		0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f,// bottom-left
		0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 1.0f, 0.0f, // bottom-left
		-0.5f, -0.5f, 0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 0.0f, // bottom-right
		-0.5f, -0.5f, -0.5f, 0.0f, -1.0f, 0.0f, 0.0f, 1.0f, // top-right
		// Top face
		-0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,// top-left
		0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom-right
		0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 1.0f, // top-right     
		0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 1.0f, 0.0f, // bottom-right
		-0.5f, 0.5f, -0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 1.0f,// top-left
		-0.5f, 0.5f, 0.5f, 0.0f, 1.0f, 0.0f, 0.0f, 0.0f // bottom-left        
	};
	glGenVertexArrays(1, &cubeVAO);
	glGenBuffers(1, &cubeVBO);
	glBindBuffer(GL_ARRAY_BUFFER, cubeVBO);
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);
	glBindVertexArray(cubeVAO);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
	glEnableVertexAttribArray(2);
	glVertexAttribPointer(2, 2, GL_FLOAT, GL_FALSE, 8 * sizeof(GLfloat), (GLvoid*)(6 * sizeof(GLfloat)));
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
	glViewport(0, 0, 1080, 720);
	glEnable(GL_CULL_FACE);
	glEnable(GL_DEPTH_TEST);
	GameMesh* boatMesh = GameMesh::CreateMesh("Boat", "Resources/boat/boat.obj");
	GameMesh* tiger = GameMesh::CreateMesh("Tiger", "Resources/tiger/tiger.obj");
	mat4 projection = perspective(radians(45.0f), 1080.0f / 720.0f, 0.1f, 100.0f);
	mat4 view = lookAt(vec3(0.0f, 0.0f, 3.0f), vec3(0.0f, 0.0f, 0.0f), vec3(0.0f, 1.0f, 0.0f));
	mat4 model = mat4(1.0f);
	model = glm::scale(model, vec3(0.5f, 0.5f, 0.5f));
	GameShader* testShader = GameShader::CreateShader("TestShader", "shader/testVS.glsl", "shader/testFS.glsl");
	
	while (!glfwWindowShouldClose(window)) {
		//boatMesh->Draw();
		glClearColor(0.3f, 0.2f, 0.1f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
		testShader->use();
		testShader->setMat4("projection", projection);
		testShader->setMat4("view", view);
		testShader->setMat4("model", model);
		boatMesh->Draw();
		//tiger->Draw();
		//glDrawArrays(GL_TRIANGLES, 0, 36);
		//glBindVertexArray(0);
		glfwPollEvents();
		glfwSwapBuffers(window);
	}
}
#pragma endregion

#pragma region start engine
int Engine::StartEngine() {
	status = BEFORE_INIT;
	if (!glfwInit()) {
		cout << "Failed to initialize GLFW" << endl;
		return -1;
	}
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 4);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
#ifdef __APPLE__
	glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE);
#endif
	status = INIT;
	window = glfwCreateWindow(scr_width, scr_height, ("MiniUnity-" + gameName).c_str(), NULL, NULL);
	if (!window) {
		cout << "Failed to create GLFW window" << endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);
	return 1;
}
#pragma endregion

#pragma region before engine run
/// <summary>
/// 應該一個loadGame(載入遊戲),一個PreLoadGame(載入遊戲設定)。遊戲內容應該用一個類“GamePackage”完成，用json file之類的儲存每個資料。
/// 時間不夠所以沒做。直接在這裡用loadGame手動創建遊戲內容。
/// </summary>
void Engine::LoadGame() {
	LoadGameEvent.Invoke();
	GameShader* shader1 = GameShader::CreateShader("Shader1", "shader/VS1.glsl", "shader/FS1.glsl");
	GameMesh* boatMesh = GameMesh::CreateMesh("Boat", "Resources/boat/boat.obj");
	GameTexture* boatColor = GameTexture::CreatePicTexture("BoatColor", "Resources/boat/boatColor.png");
	GameTexture* boatNormal = GameTexture::CreatePicTexture("BoatNormal", "Resources/boat/boatNormal.png");
	GameTexture* boatMetallic = GameTexture::CreatePicTexture("BoatMetallic", "Resources/boat/boatMetallic.png");
	GameTexture* boatRoughness = GameTexture::CreatePicTexture("BoatRoughness", "Resources/boat/boatRoughness.png");
	GameTexture* boatAO = GameTexture::CreatePicTexture("BoatAO", "Resources/boat/boatAO.png");
	GameMaterial* boatMaterial = GameMaterial::CreateMaterial("BoatMaterial", Opaque, 0, shader1);
	boatMaterial->SetTextureSlot("diffuse", boatColor);
	boatMaterial->SetTextureSlot("normal", boatNormal);
	boatMaterial->SetTextureSlot("metallic", boatMetallic);
	boatMaterial->SetTextureSlot("roughness", boatRoughness);
	boatMaterial->SetTextureSlot("ao", boatAO); 
	GameObject* boat = new GameObject("Boat");
	MeshRendererAttribute* boatMeshRendererAtt = boat->AddAttribute<MeshRendererAttribute>();
	boatMeshRendererAtt->mesh = boatMesh;
	boatMeshRendererAtt->material = boatMaterial;
	
	GameObject* light1 = new GameObject("Light1");
	LightAttribute* light1Att = light1->AddAttribute<LightAttribute>(true, true, POINT_LIGHT, Color(0.8f,0.6f,0.3f));
	light1->transform()->SetWorldPos(vec3(1.0,1.0,1.0));

	GameObject* light2 = new GameObject("Light2");
	LightAttribute* light2Att = light2->AddAttribute<LightAttribute>(true, true, DIRECTION_LIGHT, Color::Purple);
	light2->transform()->SetWorldRotation(vec3(1.0, -1.0, -1.0));

	GameObject* camera = new GameObject("Camera");
	CameraAttribute* cameraAtt = camera->AddAttribute<CameraAttribute>();
	camera->transform()->SetWorldPos(vec3(-1.0, -1.0, 1.0));
	camera->transform()->LookAt(boat->transform()->GetWorldPos());
	cameraAtt->backgroundColor = Color(0.3f, 0.2f, 0.1f);
	
};
void Engine::printAllSettings() {
	cout << "Screen Size: " << scr_width << "x" << scr_height << endl;
	cout << "Game Name: " << gameName << endl;
	cout << "Max Light Num With Depth Map: " << lightWithDepthMap_maxNum << endl;
	cout << "Max Light Num: " << light_maxNum << endl;
	cout << "Ortho Distance: " << Ortho_Distance << endl;
	cout << "Max Frame Rate: " << MaxFrameRate << endl;
	cout << "Max Fixed Update Per Second: " << MaxFixedUpdatePerSecond << endl;
	cout << "Shadow map size = " << shadowMapTextureWidth << " x " << shadowMapTextureHeight << endl;
}
void Engine::printOpenGLInfo()
{
	const GLubyte* name = glGetString(GL_VENDOR);
	const GLubyte* renderer = glGetString(GL_RENDERER);
	const GLubyte* glversion = glGetString(GL_VERSION);
	cout << "OpenGL company: " << name << endl;
	cout << "Renderer name: " << renderer << endl;
	cout << "OpenGL version: " << glversion << endl;
}
int Engine::BeforeEngineRun() {

	if (glewInit() != GLEW_OK) {
		std::cout << "GLEW not OK." << std::endl;
		return -1;
	}
	status = BEFORE_RUN;
	printOpenGLInfo();
	printAllSettings();
	glEnable(GL_DEPTH_TEST);
	glEnable(GL_CULL_FACE);

#pragma region register input events
	glfwSetFramebufferSizeCallback(window, framebuffer_size_callback);
	glfwSetKeyCallback(window, key_callback);
	glfwSetScrollCallback(window, scroll_callback);
	glfwSetCursorPosCallback(window, cursor_position_callback);
	glfwSetMouseButtonCallback(window, mouse_button_callback);
#pragma endregion

#pragma region create built-in shaders
	//pre create built=in shaders for shadow map
	GameShader::CreateShader("OtherLightShadowMapShader", "shader/ShadowMapVS.glsl", "shader/ShadowMapFS.glsl");
	GameShader::CreateShader("PointLightShadowMapShader", "shader/ShadowCubeMapVS.glsl", "shader/ShadowCubeMapFS.glsl");
	GameShader::CreateShader("PostRenderShader","shader/PostRenderVS.glsl","shader/PostRenderFS.glsl");
#pragma endregion
	
#pragma region init the final output plane
	// init the final output plane
	GLfloat planeVertices[] = {
		// Positions    // Texture Coords
		-1.0f, 1.0f, 0.0f, 0.0f, 1.0f,
		-1.0f, -1.0f, 0.0f, 0.0f, 0.0f,
		1.0f, 1.0f, 0.0f, 1.0f, 1.0f,
		1.0f, -1.0f, 0.0f, 1.0f, 0.0f,
	};
	// Setup plane
	glGenVertexArrays(1, &quadVAO_forFinalOutput);
	glGenBuffers(1, &quadVBO_forFinalOutput);
	glBindVertexArray(quadVAO_forFinalOutput);
	glBindBuffer(GL_ARRAY_BUFFER, quadVBO_forFinalOutput);
	glBufferData(GL_ARRAY_BUFFER, sizeof(planeVertices), &planeVertices, GL_STATIC_DRAW);
	glEnableVertexAttribArray(0);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)0);
	glEnableVertexAttribArray(1);
	glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(GLfloat), (GLvoid*)(3 * sizeof(GLfloat)));
#pragma endregion

#pragma region init light shadow maps
	Light::PreInitiFramebufferAndTexture();
#pragma endregion

#pragma region Build FrameBuffer for Post Render
	//build framebuffer for post rendering
	glGenFramebuffers(1, &framebuffer_postRender);
	glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_postRender);
	glGenTextures(1, &textureColorbuffer_postRender);
	glBindTexture(GL_TEXTURE_2D, textureColorbuffer_postRender);
	glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB16F, scr_width, scr_height, 0, GL_RGB, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer_postRender, 0);
	glBindTexture(GL_TEXTURE_2D, 0);
	glGenRenderbuffers(1, &depthStencilBuffer_postRender);
	glBindRenderbuffer(GL_RENDERBUFFER, depthStencilBuffer_postRender);
	glRenderbufferStorage(GL_RENDERBUFFER, GL_DEPTH24_STENCIL8, scr_width, scr_height);
	glFramebufferRenderbuffer(GL_FRAMEBUFFER, GL_DEPTH_STENCIL_ATTACHMENT, GL_RENDERBUFFER, depthStencilBuffer_postRender);
	if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
		cout << "ERROR::Post Render Framebuffer is not complete." << endl;
		return -1;
	}
	else {
		cout << "Done generate Post Render Framebuffer." << endl;
	}
	glBindFramebuffer(GL_FRAMEBUFFER, 0);
#pragma endregion

#pragma region Build Array for Shadow Map
	//texture2d array for directional light
	glGenTextures(1, &texture2dArray_forDirectionalLight);
	glBindTexture(GL_TEXTURE_2D_ARRAY, texture2dArray_forDirectionalLight);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT, shadowMapTextureWidth, shadowMapTextureHeight, lightWithDepthMap_maxNum, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, value_ptr(vec4(1.0f)));
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

	//texture2d array for spot light
	glGenTextures(1, &texture2dArray_forSpotLight);
	glBindTexture(GL_TEXTURE_2D_ARRAY, texture2dArray_forSpotLight);
	glTexImage3D(GL_TEXTURE_2D_ARRAY, 0, GL_DEPTH_COMPONENT, shadowMapTextureWidth, shadowMapTextureHeight, lightWithDepthMap_maxNum, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameterfv(GL_TEXTURE_2D_ARRAY, GL_TEXTURE_BORDER_COLOR, value_ptr(vec4(1.0f)));
	glBindTexture(GL_TEXTURE_2D_ARRAY, 0);

	//textureCube array for point light
	glGenTextures(1, &textureCubeArray_forPointLight);
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, textureCubeArray_forPointLight);
	glTexImage3D(GL_TEXTURE_CUBE_MAP_ARRAY, 0, GL_DEPTH_COMPONENT, shadowMapTextureWidth, shadowMapTextureHeight, lightWithDepthMap_maxNum * 6, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
	glTexParameteri(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
	glTexParameterfv(GL_TEXTURE_CUBE_MAP_ARRAY, GL_TEXTURE_BORDER_COLOR, value_ptr(vec4(1.0f)));
	glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);

#pragma endregion
	
	LoadGame();
	
	return 1;
}
#pragma endregion

#pragma region run engine

void Engine::RenderFinalQuad()
{
	glBindVertexArray(quadVAO_forFinalOutput);
	glDrawArrays(GL_TRIANGLE_STRIP, 0, 4);
	glBindVertexArray(0);
}
int Engine::RunEngine() {
	
	RootObjects = GameObject::GetRootGameObjects();
	if (RootObjects == nullptr) {
		cout << "RootObjects is null" << endl;
		return -1;
	}
	status = RUNNING;
	cout << "start running" << endl;

#pragma region Get Shaders
	//get shaders
	GameShader* otherLightShadowMapShader = GameShader::FindShader("OtherLightShadowMapShader");
	GameShader* pointLightShadowMapShader = GameShader::FindShader("PointLightShadowMapShader");
	GameShader* postRenderShader = GameShader::FindShader("PostRenderShader");
	GameMesh* boat = GameMesh::FindMesh_ByName("Boat");
#pragma endregion
	
	while (!glfwWindowShouldClose(window)) {
		//記錄開始時間
		auto start = std::chrono::high_resolution_clock::now();

		//創建新幀
		CreateNewRenderFrameData();

		//處理輸入事件
		glfwPollEvents();

		//Fixed Update
		if (timeCount_forFixedUpdate.count() / 1000 >= 1 / MaxFixedUpdatePerSecond) {
			for (auto& obj : *RootObjects) GameObject::RunGameObject(obj, FIXED_UPDATE);
			timeCount_forFixedUpdate = std::chrono::milliseconds(0);
		}

		//Update
		for (auto& obj : *RootObjects) GameObject::RunGameObject(obj, UPDATE);

		//Late update: (Camera，Light, Renderer 等特殊屬性在這裡提交幀資料)
		for (auto& obj : *RootObjects) GameObject::RunGameObject(obj, LATE_UPDATE);

		RenderFrameData &frameData = tempRenderFrameData;
		
#pragma region create shadow map
		//生成Shadow map
		glViewport(0, 0, shadowMapTextureWidth, shadowMapTextureHeight);
		glEnable(GL_DEPTH_TEST);
		otherLightShadowMapShader->use();
		//平行光
		for (ShadowLightData_OtherLight &light : frameData.allShadowLights_DirectionalLight) {
			glBindFramebuffer(GL_FRAMEBUFFER, light.frameBuffer_ID);
			glClear(GL_DEPTH_BUFFER_BIT);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, light.shadowMap_ID, 0);
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
			otherLightShadowMapShader->setMat4("lightVP", light.lightVP);
			for (auto &mesh_tex : frameData.allMeshesCastingShadow) {
				if (mesh_tex.second != nullptr) { //has diffuse tex
					otherLightShadowMapShader->setInt("hasDiffuseTex", 1);
					GameTexture::BindTexture(0, mesh_tex.second->GetTexID(), vec2(0.0, 0.0), "diffuseTex", *otherLightShadowMapShader);
				}
				else {
					otherLightShadowMapShader->setInt("hasDiffuseTex", 0);
				}
				otherLightShadowMapShader->setMat4("modelMatrix", mesh_tex.first.modelMatrix);
				mesh_tex.first.mesh->Draw();
			}
		}
		// 聚光燈
		for (ShadowLightData_OtherLight& light : frameData.allShadowLights_SpotLight) {
			glBindFramebuffer(GL_FRAMEBUFFER, light.frameBuffer_ID);
			glClear(GL_DEPTH_BUFFER_BIT);
			glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, light.shadowMap_ID, 0);
			glDrawBuffer(GL_NONE);
			glReadBuffer(GL_NONE);
			otherLightShadowMapShader->setMat4("lightVP", light.lightVP);
			for (auto& mesh_tex : frameData.allMeshesCastingShadow) {
				if (mesh_tex.second != nullptr) {
					otherLightShadowMapShader->setInt("hasDiffuseTex", 1);
					GameTexture::BindTexture(0, mesh_tex.second->GetTexID(), vec2(0.0, 0.0), "diffuseTex", *otherLightShadowMapShader);
				}
				else
					otherLightShadowMapShader->setInt("hasDiffuseTex", 0);
				otherLightShadowMapShader->setMat4("modelMatrix", mesh_tex.first.modelMatrix);
				mesh_tex.first.mesh->Draw();
			}
		}
		//點光
		pointLightShadowMapShader->use();
		for (ShadowLightData_PointLight &light : frameData.allShadowLights_PointLight) {
			glBindFramebuffer(GL_FRAMEBUFFER, light.frameBuffer_ID);
			pointLightShadowMapShader->setVec3("lightPos", light.lightPos);
			pointLightShadowMapShader->setFloat("farPlane", light.farPlane);
			for (int i = 0; i < 6; i++) {
				glClear(GL_DEPTH_BUFFER_BIT);
				glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, light.shadowMap_ID, 0);
				glDrawBuffer(GL_NONE);
				glReadBuffer(GL_NONE);
				mat4 lightVP = light.lightProj * light.lightViews[i];
				pointLightShadowMapShader->setMat4("lightVP", lightVP);
				for (auto &mesh_tex : frameData.allMeshesCastingShadow) {
					if (mesh_tex.second != nullptr) {
						pointLightShadowMapShader->setInt("hasDiffuseTex", 1);
						GameTexture::BindTexture(0, mesh_tex.second->GetTexID(), vec2(0.0, 0.0), "diffuseTex", *pointLightShadowMapShader);
					}
					else
						pointLightShadowMapShader->setInt("hasDiffuseTex", 0);
					pointLightShadowMapShader->setMat4("modelMatrix", mesh_tex.first.modelMatrix);
					mesh_tex.first.mesh->Draw();
				}
			}
		}

		//update each texture to array texture
		glBindTexture(GL_TEXTURE_2D_ARRAY, texture2dArray_forDirectionalLight);
		for (int i = 0; i < frameData.allShadowLights_DirectionalLight.size(); i++) {
			glCopyImageSubData(frameData.allShadowLights_DirectionalLight[i].shadowMap_ID, GL_TEXTURE_2D, 0, 0, 0, 0, texture2dArray_forDirectionalLight, GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, shadowMapTextureWidth, shadowMapTextureHeight, 1);
		}
		glBindTexture(GL_TEXTURE_2D_ARRAY, texture2dArray_forSpotLight);
		for (int i = 0; i < frameData.allShadowLights_SpotLight.size(); i++) {
			glCopyImageSubData(frameData.allShadowLights_SpotLight[i].shadowMap_ID, GL_TEXTURE_2D, 0, 0, 0, 0, texture2dArray_forSpotLight, GL_TEXTURE_2D_ARRAY, 0, 0, 0, i, shadowMapTextureWidth, shadowMapTextureHeight, 1);
		}
		glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, textureCubeArray_forPointLight);
		for (int i = 0; i < frameData.allShadowLights_PointLight.size(); i++) {
			for (int j = 0; j < 6; j++) {
				glCopyImageSubData(frameData.allShadowLights_PointLight[i].shadowMap_ID, GL_TEXTURE_CUBE_MAP, 0, 0, 0, j, textureCubeArray_forPointLight, GL_TEXTURE_CUBE_MAP_ARRAY, 0, 0, 0, i * 6 + j, shadowMapTextureWidth, shadowMapTextureHeight, 1);
			}
		}
		glBindTexture(GL_TEXTURE_2D_ARRAY, 0);
		glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, 0);
		glBindFramebuffer(GL_FRAMEBUFFER, 0);
#pragma endregion

		//更新logic frame提交的light data SSBO 和camera UBO
		Camera::UpdateUBO(frameData.cameraVPData);
		Light::SetAllSSBOData(frameData.allSingleLights, frameData.ambientLight, frameData.ambientPower);
		
#pragma region Render
		//正式渲染
		glReadBuffer(GL_FRONT);
		glDrawBuffer(GL_BACK);
		glViewport(0, 0, scr_width, scr_height);
		//glBindFramebuffer(GL_FRAMEBUFFER, framebuffer_postRender);
		//glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer_postRender, 0);
		//glDrawBuffer(GL_COLOR_ATTACHMENT0);
		//glReadBuffer(GL_NONE);
		Color bgColor = frameData.cameraBackgroundColor;
		glClearColor(bgColor.r, bgColor.g, bgColor.b, bgColor.a);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); //clear framebuffer
		glEnable(GL_DEPTH_TEST);
		
		GameShader* lastShader = nullptr;
		for (auto& renderFunc_shader : frameData.opaqueRenderFuncs) {
			if (renderFunc_shader.second == nullptr) continue;
			GameShader* thisShader = renderFunc_shader.second;

			//if shader not be same, need to transfer all variables & data to shader again
			if (lastShader != thisShader) {
				thisShader->use();
				lastShader = thisShader;

				//send texture2d array sampler to shader
				currentTextureBindingPosition = 0;
				glActiveTexture(GL_TEXTURE0+ currentTextureBindingPosition);
				glBindTexture(GL_TEXTURE_2D_ARRAY, texture2dArray_forDirectionalLight);
				thisShader->setInt("shadowMapTexture2DArray_DirectionalLight", currentTextureBindingPosition);
				thisShader->setInt("DirectionalLightShadowCount", frameData.allShadowLights_DirectionalLight.size());
				for (int i = 0; i < frameData.allShadowLights_DirectionalLight.size(); i++) {
					string vpName = "DirectLightVP_Bias[" + to_string(i) + "]";
					mat4 m = shadowMap_biasMatrix * frameData.allShadowLights_DirectionalLight[i].lightVP;
					thisShader->setMat4(vpName, m);
					string posName = "DirectionalLightDirections[" + to_string(i) + "]";
					thisShader->setVec3(posName, frameData.allShadowLights_DirectionalLight[i].lightPos_OR_lightDir);
				}

				currentTextureBindingPosition++;
				glActiveTexture(GL_TEXTURE0+ currentTextureBindingPosition);
				glBindTexture(GL_TEXTURE_2D_ARRAY, texture2dArray_forSpotLight);
				thisShader->setInt("shadowMapTexture2DArray_SpotLight", currentTextureBindingPosition);
				thisShader->setInt("SpotLightShadowCount", frameData.allShadowLights_SpotLight.size());
				for (int i = 0; i < frameData.allShadowLights_SpotLight.size(); i++) {
					string vpName = "SpotLightVP_Bias[" + to_string(i) + "]";
					mat4 m = shadowMap_biasMatrix * frameData.allShadowLights_SpotLight[i].lightVP;
					thisShader->setMat4(vpName, m);
					string posName = "SpotLightPositions[" + to_string(i) + "]";
					thisShader->setVec3(posName, frameData.allShadowLights_SpotLight[i].lightPos_OR_lightDir);
				}

				currentTextureBindingPosition++;
				glActiveTexture(GL_TEXTURE0+ currentTextureBindingPosition);
				glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, textureCubeArray_forPointLight);
				thisShader->setInt("shadowMapTextureCubeArray_PointLight", currentTextureBindingPosition);
				thisShader->setInt("PointLightShadowCount", frameData.allShadowLights_PointLight.size());
				for (int i = 0; i < frameData.allShadowLights_PointLight.size(); i++) {
					string pName = "PointLightPositions[" + to_string(i) + "]";
					thisShader->setVec3(pName, frameData.allShadowLights_PointLight[i].lightPos);
					string farPlaneName = "PointLightFarPlanes[" + to_string(i) + "]";
					thisShader->setFloat(farPlaneName, frameData.allShadowLights_PointLight[i].farPlane);
				}
			}
			renderFunc_shader.first(); //draw function, including binding other textures, variables, modelMatrix, etc.
		}
		for (auto& order_renderFunc_shader : frameData.transparentRenderFuncs) {
			if (order_renderFunc_shader.second.second == nullptr) continue;
			GameShader* thisShader = order_renderFunc_shader.second.second;
			if (lastShader != thisShader) {
				thisShader->use();
				lastShader = thisShader;

				//send texture2d array sampler to shader
				currentTextureBindingPosition = 0;
				glActiveTexture(GL_TEXTURE0 + currentTextureBindingPosition);
				glBindTexture(GL_TEXTURE_2D_ARRAY, texture2dArray_forDirectionalLight);
				thisShader->setInt("shadowMapTexture2DArray_DirectionalLight", currentTextureBindingPosition);
				thisShader->setInt("DirectionalLightShadowCount", frameData.allShadowLights_DirectionalLight.size());
				for (int i = 0; i < frameData.allShadowLights_DirectionalLight.size(); i++) {
					string vpName = "DirectLightVP_Bias[" + to_string(i) + "]";
					mat4 m = shadowMap_biasMatrix * frameData.allShadowLights_DirectionalLight[i].lightVP;
					thisShader->setMat4(vpName, m);
				}

				currentTextureBindingPosition++;
				glActiveTexture(GL_TEXTURE0 + currentTextureBindingPosition);
				glBindTexture(GL_TEXTURE_2D_ARRAY, texture2dArray_forSpotLight);
				thisShader->setInt("shadowMapTexture2DArray_SpotLight", currentTextureBindingPosition);
				thisShader->setInt("SpotLightShadowCount", frameData.allShadowLights_SpotLight.size());
				for (int i = 0; i < frameData.allShadowLights_SpotLight.size(); i++) {
					string vpName = "SpotLightVP_Bias[" + to_string(i) + "]";
					mat4 m = shadowMap_biasMatrix * frameData.allShadowLights_SpotLight[i].lightVP;
					thisShader->setMat4(vpName, m);
				}

				currentTextureBindingPosition++;
				glActiveTexture(GL_TEXTURE0 + currentTextureBindingPosition);
				glBindTexture(GL_TEXTURE_CUBE_MAP_ARRAY, textureCubeArray_forPointLight);
				thisShader->setInt("shadowMapTextureCubeArray_PointLight", currentTextureBindingPosition);
				thisShader->setInt("PointLightShadowCount", frameData.allShadowLights_PointLight.size());
				for (int i = 0; i < frameData.allShadowLights_PointLight.size(); i++) {
					string pName = "PointLightPositions[" + to_string(i) + "]";
					thisShader->setVec3(pName, frameData.allShadowLights_PointLight[i].lightPos);
				}

			}
			order_renderFunc_shader.second.first();
		}
		for (auto& renderFunc_shader : frameData.uiRenderFuncs) {
			renderFunc_shader.first();
		}
		
#pragma endregion

#pragma region Post Render
		/**
		//後處理
		glDisable(GL_DEPTH_TEST);
		glBindFramebuffer(GL_FRAMEBUFFER, 0); // to default framebuffer
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT | GL_STENCIL_BUFFER_BIT); //clear framebuffer
		glClearColor(1.0f, 1.0f, 1.0f, 1.0f);
		glDrawBuffer(GL_BACK);
		glReadBuffer(GL_FRONT);

		//bind texture
		glActiveTexture(GL_TEXTURE0);
		glBindTexture(GL_TEXTURE_2D, textureColorbuffer_postRender);
		glActiveTexture(GL_TEXTURE1);
		glBindTexture(GL_TEXTURE_2D, depthStencilBuffer_postRender);

		postRenderShader->use();
		postRenderShader->setInt("screenColorTexture", 0);
		postRenderShader->setInt("depthStencilTexture", 1);
		RenderFinalQuad();
		**/
#pragma endregion

		glfwSwapBuffers(window);

		//記錄結束時間, 計算時間差
		auto end = std::chrono::high_resolution_clock::now();
		deltaTime = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		timeCount_forFixedUpdate += deltaTime;
		if ((deltaTime.count() / 1000) < (1.0 / MaxFrameRate)) {
			std::this_thread::sleep_for(std::chrono::milliseconds((int)((1.0 / MaxFrameRate - deltaTime.count() / 1000) * 1000)));
		}
		
	}
#pragma region Delete all

	Camera::DeleteCameraUBO();
	Light::DeleteLightSSBO();
	for (auto& obj : *GameObject::GetRootGameObjects()) {
		obj->Destroy(); //will auto remove lights frameBuffer, shadowMap, ...
	}
	glDeleteTextures(1, &texture2dArray_forDirectionalLight);
	glDeleteTextures(1, &texture2dArray_forSpotLight);
	glDeleteTextures(1, &textureCubeArray_forPointLight);
	glDeleteTextures(1, &textureColorbuffer_postRender);
	glDeleteFramebuffers(1, &framebuffer_postRender);
	glDeleteRenderbuffers(1, &depthStencilBuffer_postRender);

#pragma endregion
	return 1;
}
#pragma endregion

#pragma region end engine
//end engine
void Engine::EndEngine() {
	glfwTerminate();
}
#pragma endregion

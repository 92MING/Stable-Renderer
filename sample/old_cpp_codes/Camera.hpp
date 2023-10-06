//包括相機及相機Attribute
#pragma once
#include "Dependencies/glew/glew.h"
#include "Dependencies/glm/gtc/matrix_transform.hpp"
#include "DataStructure.hpp"
#include "GameCore.hpp"
#include "Engine.h"

#pragma region Camera
class Camera {
protected:
	static Camera* mainCam;
	static unordered_set<Camera*> AllCameras;
	static vector<Camera*> ActiveCameras;
	//for UBO
	static bool hasInitUBO;
	static GLuint cameraUBO;
	
	void ChangeCameraActiveState(bool active){
		if (active && !isActiveCamera()) {
			ActiveCameras.push_back(this);
			if (ActiveCameras.size() == 1) mainCam = this; // set this camera as main camera if there is no active camera before (this is the first active camera
		}
		else if (isActiveCamera()){
			remove(ActiveCameras.begin(), ActiveCameras.end(), this);
			if (mainCam == this) {
				if (ActiveCameras.size() > 0) mainCam = *ActiveCameras.begin(); // set the first active camera as main camera
				else mainCam = nullptr;
			}
		}
	}
	Camera() {
		AllCameras.insert(this);
	}
	~Camera() {
		AllCameras.erase(this);
		ChangeCameraActiveState(false);
	}
public:
	float fov = 90.0f;
	float nearPlane = 0.1f;
	float farPlane = 100.0f;
	float orthoSize = 1.0f; //for orthogonal projection
	Color backgroundColor = Color::Black; 
	CameraProjectionType projectionType = Perspective;
	
	bool isMainCam() {
		return mainCam == (Camera*)this;
	}
	bool isActiveCamera(){
		return find(ActiveCameras.begin(), ActiveCameras.end(), this) != ActiveCameras.end();
	}
	void SetAsMainCam() {
		if (!isActiveCamera()) return;
		mainCam = this;
	}
	virtual vec3 GetPostion() { return vec3(0.0,0.0,0.0); }
	virtual vec3 GetForward() { return vec3(1.0, 0.0, 0.0); }
	virtual vec3 GetUp() { return vec3(0.0,1.0,0.0); };
	pair<float,float> GetSceneWidthAndHeight_NearPlane() {
		float width = 2 * tan(fov / 2) * nearPlane;
		float height = width / Engine::GetAspectRatio();
		return pair<float, float>(width, height);
	}
	mat4 GetViewMatrix() {
		return lookAt(GetPostion(), GetPostion() + GetForward(), GetUp());
	}
	mat4 GetProjectionMatrix() const {
		if (projectionType == Perspective) {
			return perspective(radians(fov), (float)Engine::GetWindowWidth() /Engine::GetWindowHeight(), nearPlane, farPlane);
		}
		else if (projectionType == Orthographic) {
			float screenScale = (float)Engine::GetWindowWidth() / Engine::GetWindowHeight();
			screenScale *= orthoSize;
			screenScale /= 2;
			return ortho<float>(-screenScale * Engine::GetOrthoDistance(), screenScale* Engine::GetOrthoDistance(), -(float)Engine::GetOrthoDistance(), (float)Engine::GetOrthoDistance(), nearPlane, farPlane);
		}
	}	
	
	static void InitCameraUBO() {
		if (hasInitUBO) return;
		glGenBuffers(1, &cameraUBO);
		glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
		glBufferData(GL_UNIFORM_BUFFER, sizeof(CameraVPData), NULL, GL_DYNAMIC_DRAW);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
		glBindBufferBase(GL_UNIFORM_BUFFER, 0, cameraUBO);
		hasInitUBO = true;
	}
	//call before render.
	static void UpdateUBO(CameraVPData data) {
		auto UBO_VPData = data;
		if (!hasInitUBO) InitCameraUBO();
		glBindBuffer(GL_UNIFORM_BUFFER, cameraUBO);
		glBufferSubData(GL_UNIFORM_BUFFER, 0, sizeof(CameraVPData), &UBO_VPData);
		glBindBuffer(GL_UNIFORM_BUFFER, 0);
	}
	static void DeleteCameraUBO() {
		if (!hasInitUBO) return;
		glDeleteBuffers(1, &cameraUBO);
		hasInitUBO = false;
	}
	static GLuint GetCameraUBO_ID() {
		if (!hasInitUBO) InitCameraUBO();
		return cameraUBO;
	}
	static Camera* GetMainCam() {
		return mainCam;
	}
	static bool hasMainCam() { return !(mainCam == nullptr); }
};
bool Camera::hasInitUBO = false;
Camera* Camera::mainCam;
unordered_set<Camera*> Camera::AllCameras;
vector<Camera*> Camera::ActiveCameras;
GLuint Camera::cameraUBO;

#pragma endregion

#pragma region CameraAttribute

char cameraAttributeTypeName[] = "CameraAttribute";
class CameraAttribute :public GameAttribute<CameraAttribute, cameraAttributeTypeName>, public Camera {
public:
	CameraAttribute(GameObject* gameObject, bool enable) :GameAttribute(gameObject, enable) {}
	virtual vec3 GetPostion() override {
		return this->gameObject->transform()->GetWorldPos();
	};
	virtual vec3 GetForward() override {
		return this->gameObject->transform()->forward();
	};
	virtual vec3 GetUp() override {
		return this->gameObject->transform()->up();
	};
	void OnEnable() override {
		ChangeCameraActiveState(true); //設置為啟用中的相機
	}
	void OnDisable() override {
		ChangeCameraActiveState(false); //設置為禁用中的相機
	}
	void LateUpdate() override {
		//如果是主相機，則在邏輯幀結束的時候向Engine提交這一幀的相機渲染數據
		if (isMainCam()) Engine::SetCameraDataToRenderFrameData(CameraVPData{ GetPostion() ,GetViewMatrix(), GetProjectionMatrix()}, backgroundColor);
		
	}
};
#pragma endregion
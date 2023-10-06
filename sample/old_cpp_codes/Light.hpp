#pragma once
#include "Camera.hpp"
using namespace std;
using namespace glm;
#define UNCHANGE FLT_MIN

#pragma region Light
class Light {
protected:
    
    static bool hasInitSSBO;
	static GLuint lightSSBO_ID;
	static Color ambientLightColor;
	static float ambientLightIntensity;
	static unique_list<Light*> lightsEnable;  //enable lights
    static unique_list<Light*> lightsCastingShadow; //lights casting shadow(need FBO)
    
    //common
	SingleLightData thisLightData;
    void setLightEnable(bool set) {
        //unique_list已經避免重複
		if (set) lightsEnable.push_back(this);
		else lightsEnable.remove(this);
    }

    //for shadow
    bool castShadow = false;
    static bool hasInitFrameBufferAndTexture;
	static vector<FBO_Texture> allPointLightShadowFBOandTex;
	static vector<FBO_Texture> allOtherLightShadowFBOandTex;
    GLuint frameBuffer_ID; //深度幀緩衝
	GLuint shadowMap_ID; //深度貼圖(cube map for point light)
    float farPlane_pointLight = 100.0;
    
    Light(bool enable=true, bool castShadow = false, LightType lightType = POINT_LIGHT, Color lightColor = Color::White, float intensity = 1.0f){
        setLightEnable(enable);
        if (enable) lightsEnable.push_back(this);
        thisLightData.lightType = lightType;
        thisLightData.lightColor = lightColor;
        thisLightData.intensity = intensity;
        if (castShadow) SetCastShadow(true);
	}
    ~Light() {
        lightsEnable.remove(this); //if not enable, this do nothing
        if (castShadow) SetCastShadow(false);
    }
    //初始化光線SSBO
    static void InitSSBO() {
        if (hasInitSSBO) return;
        glGenBuffers(1, &lightSSBO_ID);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightSSBO_ID);
        glBufferData(GL_SHADER_STORAGE_BUFFER, sizeof(float)+sizeof(vec3)+sizeof(unsigned int)+sizeof(SingleLightData)*Engine::GetMaxLightNum(), NULL, GL_DYNAMIC_DRAW);
        glBindBufferBase(GL_SHADER_STORAGE_BUFFER, 1, lightSSBO_ID);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
        hasInitSSBO = true;
    }
public:
    //設置是否使用深度貼圖(是否造成陰影)
    static void PreInitiFramebufferAndTexture() {
        if (!hasInitFrameBufferAndTexture) {
            //init point lights
            vec2 mapSize = Engine::GetShadowMapTextureSize();
			for (int i = 0; i < Engine::GetMaxLightNumWithDepthMap(); i++) {
                FBO_Texture fboTex;
                glGenFramebuffers(1, &fboTex.FBO_ID);
                glBindFramebuffer(GL_FRAMEBUFFER, fboTex.FBO_ID);
                glGenTextures(1, &fboTex.shadowMapTexture_ID);//創建深度貼圖
                glBindTexture(GL_TEXTURE_CUBE_MAP, fboTex.shadowMapTexture_ID);
                for (GLuint i = 0; i < 6; ++i) {
                    glTexImage2D(GL_TEXTURE_CUBE_MAP_POSITIVE_X + i, 0, GL_DEPTH_COMPONENT, mapSize.x, mapSize.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
                    glTexParameteri(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_WRAP_R, GL_CLAMP_TO_BORDER);
                    glTexParameterfv(GL_TEXTURE_CUBE_MAP, GL_TEXTURE_BORDER_COLOR, value_ptr(vec4(1.0f)));
                }
                glFramebufferTexture(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, fboTex.shadowMapTexture_ID, 0);
                if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
                    cout << "ERROR when generating light shadow map frambuffer: not complete." << endl;
                    return;
                }
                glDrawBuffer(GL_NONE);
                glReadBuffer(GL_NONE);
                glBindTexture(GL_TEXTURE_CUBE_MAP, 0);
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
                allPointLightShadowFBOandTex.push_back(fboTex);
			}
			//init other lights
			for (int i = 0; i < Engine::GetMaxLightNumWithDepthMap(); i++) {
				FBO_Texture fboTex;
				glGenFramebuffers(1, &fboTex.FBO_ID);
				glBindFramebuffer(GL_FRAMEBUFFER, fboTex.FBO_ID);
				glGenTextures(1, &fboTex.shadowMapTexture_ID);//創建深度貼圖
				glBindTexture(GL_TEXTURE_2D, fboTex.shadowMapTexture_ID);
				glTexImage2D(GL_TEXTURE_2D, 0, GL_DEPTH_COMPONENT, mapSize.x, mapSize.y, 0, GL_DEPTH_COMPONENT, GL_FLOAT, NULL);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
				glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
				glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, value_ptr(vec4(1.0f)));
                glFramebufferTexture2D(GL_FRAMEBUFFER, GL_DEPTH_ATTACHMENT, GL_TEXTURE_2D, fboTex.shadowMapTexture_ID, 0);
				glDrawBuffer(GL_NONE);
				glReadBuffer(GL_NONE);
				if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE) {
					cout << "ERROR when generating light shadow map frambuffer: not complete." << endl;
					return;
				}
				glBindTexture(GL_TEXTURE_2D, 0);
				glBindFramebuffer(GL_FRAMEBUFFER, 0);
                allOtherLightShadowFBOandTex.push_back(fboTex);
			}
            cout << "done generate all light shadow map and texture" << endl;
            hasInitFrameBufferAndTexture = true;
        }
    }
    void SetCastShadow(bool castShadow) {
		if (this->castShadow == castShadow) return;
        if (castShadow) {
            if (this->thisLightData.lightType == POINT_LIGHT) {
                for (auto& fboTex : allPointLightShadowFBOandTex) {
                    if (!fboTex.occupied) {
						this->frameBuffer_ID = fboTex.FBO_ID;
						this->shadowMap_ID = fboTex.shadowMapTexture_ID;
						fboTex.occupied = true;
                        lightsCastingShadow.push_back(this);
                        this->castShadow = true;
                        return;
                    }
                }
                cout << "no free point light shadow map for using" << endl;
                return;
            }
            else {
                for (auto& fboTex : allOtherLightShadowFBOandTex) {
					if (!fboTex.occupied) {
						this->frameBuffer_ID = fboTex.FBO_ID;
						this->shadowMap_ID = fboTex.shadowMapTexture_ID;
						fboTex.occupied = true;
                        lightsCastingShadow.push_back(this);
                        this->castShadow = true;
						return;
					}
                }
                cout << "no free other light shadow map for using " << endl;
                return;
            }
        }
        else {
            lightsCastingShadow.remove(this);
            if (this->thisLightData.lightType == POINT_LIGHT) {
                for (auto& fboTex : allPointLightShadowFBOandTex) {
                    if (fboTex.FBO_ID == this->frameBuffer_ID) {
                        fboTex.occupied = false;
                        this->frameBuffer_ID = NULL;
                        this->shadowMap_ID = NULL;
                        return;
                    }
                }
            }
            else {
                for (auto& fboTex : allOtherLightShadowFBOandTex) {
                    if (fboTex.FBO_ID == this->frameBuffer_ID) {
                        fboTex.occupied = false;
                        this->frameBuffer_ID = NULL;
                        this->shadowMap_ID = NULL;
                        return;
                    }
                }
            }
            glDeleteTextures(1, &shadowMap_ID);
        }
    }
    SingleLightData GetThisLightData() const{
        return thisLightData;
    }
    void SetLightType(LightType type) {
        if (thisLightData.lightType == type) return;
        if (castShadow) {
			SetCastShadow(false);
			thisLightData.lightType = type;
			SetCastShadow(true);
		}
		else {
			thisLightData.lightType = type;
        }
	}
	void SetSpotLightCutOff(float newCutOff=-1, float newOuterCutOff=-1) {
        if (thisLightData.lightType != LightType::SPOT_LIGHT) {
            cout<< "This light is not a spot light.No need to set spot light cutoff."<<endl;
            return;
        }
        if (newCutOff >=0) thisLightData.cutOff = newCutOff;
        if (newOuterCutOff >=0) thisLightData.outerCutOff = newOuterCutOff;
	}
    //Distance = 1/(constant + linear * distance + quadratic * (distance * distance)), 影響光照強度隨距離的變化
	void SetDistanceFormula(float quadratic= UNCHANGE, float linear= UNCHANGE, float constant= UNCHANGE) {
        if (thisLightData.lightType != LightType::POINT_LIGHT && thisLightData.lightType != LightType::SPOT_LIGHT) {
            cout<< "This light is not a point light or spot light.No need to set distance formula."<<endl;
            return;
        }
        if (quadratic != UNCHANGE) thisLightData.quadratic = quadratic;
        if (linear != UNCHANGE) thisLightData.linear = linear;
        if (constant != UNCHANGE) thisLightData.constant = constant;
	}
	void SetLightColor(Color color) {
        thisLightData.lightColor = color;
	}
    void SetLightIntensity(float intensity) {
        if (intensity<0) {
            cout<< "Light intensity cannot be negative."<<endl;
            return;
        }
        thisLightData.intensity = intensity;
    }
    void SetPointLightShadowMapFarPlane(float farPlane) {
        if (this->thisLightData.lightType != POINT_LIGHT) {
			cout << "This light is not a point light.No need to set point light shadow map far plane." << endl;
            return;
        }
		else if (farPlane < 0) {
			cout << "Point light shadow map far plane cannot be negative." << endl;
			return;
		}
		this->farPlane_pointLight = farPlane;
    }
    bool isShadowLight() {
		return castShadow;
    }
    static void SetAmbientLight(Color color, float intensity=UNCHANGE) {
		ambientLightColor = color;
		if (intensity!=UNCHANGE) ambientLightIntensity = intensity;
    }
    
    ShadowLightData_OtherLight GetThisShadowLightData_OtherLight() const {
        if (!castShadow || thisLightData.lightType == POINT_LIGHT) return ShadowLightData_OtherLight();
        ShadowLightData_OtherLight shadowLightData;
        shadowLightData.frameBuffer_ID = this->frameBuffer_ID;
        shadowLightData.shadowMap_ID = this->shadowMap_ID;
		shadowLightData.lightVP = mat4(1.0f);
        shadowLightData.lightType = this->thisLightData.lightType;
		if (this->thisLightData.lightType == SPOT_LIGHT) shadowLightData.lightPos_OR_lightDir = this->thisLightData.position;
		else if (this->thisLightData.lightType == DIRECTION_LIGHT) shadowLightData.lightPos_OR_lightDir = this->thisLightData.direction;
		if (!Camera::hasMainCam()) return shadowLightData;
        auto cam = Camera::GetMainCam();
        mat4 v = mat4(1.0);
        mat4 p = mat4(1.0);
		if (thisLightData.lightType == LightType::DIRECTION_LIGHT) {
            float scale = (Engine::GetShadowMapTextureSize().x / (float)Engine::GetShadowMapTextureSize().y) / 2;
            p = ortho<float>((-scale) * Engine::GetOrthoDistance(), scale * Engine::GetOrthoDistance(), (- scale) * Engine::GetOrthoDistance(), scale* Engine::GetOrthoDistance(), cam->nearPlane, cam->farPlane);
            lookAt(-10.0f * thisLightData.direction, vec3(0.0f, 0.0f, 0.0f), cam->GetUp());
        }
		else if (thisLightData.lightType == LightType::SPOT_LIGHT) {
			p = perspective(radians(thisLightData.outerCutOff * 2), Engine::GetShadowMapTextureSize().x / (float)Engine::GetShadowMapTextureSize().y, cam->nearPlane, cam->farPlane);
			v = lookAt(thisLightData.position, thisLightData.position + thisLightData.direction, cam->GetUp());
        }
        shadowLightData.lightVP = p * v;
		return shadowLightData;
	}
    ShadowLightData_PointLight GetThisShadowLightData_PointLight() const {
        if (!castShadow || thisLightData.lightType != POINT_LIGHT) return ShadowLightData_PointLight();
        ShadowLightData_PointLight shadowLightData;
        shadowLightData.frameBuffer_ID = this->frameBuffer_ID;
        shadowLightData.shadowMap_ID = this->shadowMap_ID;
        shadowLightData.lightPos = this->thisLightData.position;
        shadowLightData.farPlane = this->farPlane_pointLight;
        shadowLightData.lightProj = perspective(radians(90.0f), Engine::GetShadowMapTextureSize().x /(float)Engine::GetShadowMapTextureSize().y,0.0f,farPlane_pointLight);
		shadowLightData.lightViews[0] = lookAt(thisLightData.position, thisLightData.position + vec3(1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f));
        shadowLightData.lightViews[1] = lookAt(thisLightData.position, thisLightData.position + vec3(-1.0f, 0.0f, 0.0f), vec3(0.0f, -1.0f, 0.0f));
        shadowLightData.lightViews[2] = lookAt(thisLightData.position, thisLightData.position + vec3(0.0f, 1.0f, 0.0f), vec3(0.0f, 0.0f, 1.0f));
        shadowLightData.lightViews[3] = lookAt(thisLightData.position, thisLightData.position + vec3(0.0f, -1.0f, 0.0f), vec3(0.0f, 0.0f, -1.0f));
        shadowLightData.lightViews[4] = lookAt(thisLightData.position, thisLightData.position + vec3(0.0f, 0.0f, 1.0f), vec3(0.0f, -1.0f, 0.0f));
        shadowLightData.lightViews[5] = lookAt(thisLightData.position, thisLightData.position + vec3(0.0f, 0.0f, -1.0f), vec3(0.0f, -1.0f, 0.0f));
        return shadowLightData;
    }
	static unique_list<Light*>* GetLightsCastingShadow() {
		return &lightsCastingShadow;
	}
	static unique_list<Light*>* GetLightsEnable() {
		return &lightsEnable;
	}
    static int currentEnableLightNum() {
		return lightsEnable.size();
	}
	static int currentLightUsingDepthMapNum() {
		return lightsCastingShadow.size();
    }
    static float GetAmbientLightPower() {
        return ambientLightIntensity;
    }
    static Color GetAmbientLightColor() {
        return ambientLightColor;
    }
    static GLuint GetLightSSBO_ID() {
		if (!hasInitSSBO) {
            InitSSBO();
		}
		return lightSSBO_ID;
    }
    
    //設置光線SSBO資料, call before render
	static void SetAllSSBOData(vector<SingleLightData> allLights, Color ambientLight, float ambientPower) {
        LightData_SSBO lightSSBOData;
        lightSSBOData.allSingleLightData = allLights;
		lightSSBOData.singleLightDataLength = allLights.size();
		lightSSBOData.ambientLight = ambientLight;
		lightSSBOData.ambientPower = ambientPower;
        if (!hasInitSSBO) InitSSBO();
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, lightSSBO_ID);
        glBufferSubData(GL_SHADER_STORAGE_BUFFER, 0, sizeof(float) + sizeof(vec3) + sizeof(unsigned int) + sizeof(SingleLightData) * allLights.size(), &lightSSBOData);
        glBindBuffer(GL_SHADER_STORAGE_BUFFER, 0);
    }
    static void DeleteLightSSBO() {
		if (hasInitSSBO) {
			glDeleteBuffers(1, &lightSSBO_ID);
			hasInitSSBO = false;
		}
    }
};
bool Light::hasInitSSBO = false;
unique_list<Light*> Light::lightsCastingShadow;
unique_list<Light*> Light::lightsEnable;
GLuint Light::lightSSBO_ID;
Color Light::ambientLightColor = Color::Black;
float Light::ambientLightIntensity = 1.0f;
bool Light::hasInitFrameBufferAndTexture = false;
vector<FBO_Texture> Light::allPointLightShadowFBOandTex;
vector<FBO_Texture> Light::allOtherLightShadowFBOandTex;

#pragma endregion

#pragma region LightAttribute

char LightAttributeTypeName[] = "LightAttribute";
class LightAttribute : public Light, public GameAttribute<LightAttribute, LightAttributeTypeName> {
private:
    static bool submittedAmbient; //clear flag in update(), and set true in lateUpdate()
public:
    void OnEnable() override { 
        //不會重複，因為lightsEnable是unique_list
        setEnable(true);
    }
	void OnDisable() override {
        setEnable(false);
	}
    LightAttribute(GameObject* gameObject, bool enable, bool castShadow = false, LightType lightType = POINT_LIGHT,
                   Color lightColor = Color::White, float intensity = 1.0) 
        : Light(enable,castShadow, lightType, lightColor, intensity), GameAttribute(gameObject, enable) {}
    
    void Update() override {
		submittedAmbient = false; //clear flag
    }
    void LateUpdate() override {
        //Ambinent light(submit one time only)
		if (!submittedAmbient) {
			Engine::SetAmbientLightDataToRenderFrameData(ambientLightColor, ambientLightIntensity);
			submittedAmbient = true;
		}

        //single light
        thisLightData.position = gameObject->transform()->GetWorldPos();
        thisLightData.direction = gameObject->transform()->forward();
		Engine::AddSingleLightDataToRenderFrameData(thisLightData);
        
        //shadow light
		if (castShadow) {
            if (thisLightData.lightType == POINT_LIGHT) Engine::AddShadowLightDataToRenderFrameData(GetThisShadowLightData_PointLight());
			else Engine::AddShadowLightDataToRenderFrameData(GetThisShadowLightData_OtherLight());
		}
    }

};
bool LightAttribute::submittedAmbient = false;

#pragma endregion
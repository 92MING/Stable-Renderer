//包含了常見的數據結構庫引用，及自定義的數據結構、enum。
#pragma once
#include "Dependencies/glm/glm.hpp"
#include "Dependencies/glew/glew.h"
#include "Color.hpp"
#include <queue>
#include <deque>
#include <unordered_set>
#include <unordered_map>
#include <set>
#include <map>
#include <list>
#include <vector>
#include <functional>
#include <string>
#include <climits>
#include <any>
using namespace std;
using namespace glm;
class GameShader;
class GameTexture;
class GameMaterial;
class GameMesh;

enum EngineRunStage {
    FIXED_UPDATE,
    UPDATE,
    LATE_UPDATE,
};
enum EngineStatus {
    BEFORE_INIT,
    INIT,
    BEFORE_RUN,
    RUNNING,
    STOP,
};
enum RenderType {
    Opaque,
    Transparent,
    UI
};
enum TextureWrapMode {
    Repeat, //重複
    Mirror_Repeat, //鏡像重複
    Clamp_Edge, //不重复，延伸邊沿
    Clamp_Boarder // 不重复。邊緣外設置為透明
};
enum ShaderUniformVariableType {
    Int,
    Float,
    Vec2,
    Vec3,
    Vec4,
    Mat3,
    Mat4,
};
enum CameraProjectionType {
    Orthographic,
    Perspective
};
enum LightType {
    DIRECTION_LIGHT,
    SPOT_LIGHT,
    POINT_LIGHT
};

//與queue一樣，但加上了遍歷
template<typename T, typename Container = deque<T> >
class iterable_queue : public queue<T, Container>{
public:
    typedef typename Container::iterator iterator;
    typedef typename Container::const_iterator const_iterator;

    iterator begin() { return this->c.begin(); }
    iterator end() { return this->c.end(); }
};

//與普通的list一樣，但不能重複，且支持以index現實在任意地方插入/刪除/修改
template <typename T>
class unique_list {
private:
    list<T> _list;
public:
    bool contains(T t) {
        for (auto i : _list) if (i == t) return true;
        return false;
    }
    int FindIndex(T t) {
        int index = 0;
        for (auto i : _list) {
            if (i == t) return index;
            index++;
        }
        return -1;
    }
    void push_back(T t) {
        if (!contains(t)) _list.push_back(t);
    }
    void push_front(T t) {
        if (!contains(t)) _list.push_front(t);
    }
    void pop_back() {
        _list.pop_back();
    }
    void pop_front() {
        _list.pop_front();
    }
    T front() {
        return _list.front();
    }
    T back() {
        return _list.back();
    }
    void removeAt(unsigned int pos) {
        if (pos >= _list.size()) return;
        auto it = next(_list.begin(), pos);
        _list.erase(it);
    }
    void remove(T t) {
        for (auto it = _list.begin(); it != _list.end(); it++) {
            if (*it == t) {
                _list.erase(it);
                return;
            }
        }
    }
    void moveElementTo(T t, unsigned int pos) {
        if (pos >= _list.size()) return;
        if (!contains(t)) return;
        remove(t);
        auto it = next(_list.begin(), pos);
        _list.insert(it, t);
    }
    void insertAt(T t, unsigned int position) {
        if (position == 0) {
            push_front(t);
            return;
        }
        if (position > _list.size()) return;
        auto temp = _list.begin();
        auto x = _list.begin();
        for (int i = 0; i < _list.size(); i++) {
            if (*temp == t) return;
            if (i == position) x = temp;
            temp = next(temp, 1);
        }
        _list.insert(x, t);
    }
    int getPosition(T t) {
        int index = 0;
        for (auto i : _list) {
            if (i == t) return index;
            index++;
        }
        return -1;
    }
    void clear() {
        _list.clear();
    }
    unsigned int size() {
        return _list.size();
    }
    void setAt(T t, unsigned int pos) {
        if (pos >= _list.size()) return;
        if (contains(t)) return;
        auto it = next(_list.begin(), pos);
        *it = t;
    }
    void setElementTo(T oldPos, T newPos) {
        int i = FindIndex(oldPos);
        if (i == -1) return;
        setAt(newPos, i);
    }
    T operator[](unsigned int pos) {
        if (pos >= _list.size()) return T();
        auto it = next(_list.begin(), pos);
        return *it;
    }
    auto begin() { return _list.begin(); }
    auto end() { return _list.end(); }
    unique_list<T> copy() {
        unique_list<T> newList;
        for (T x : _list) {
            newList.push_back(x);
        }
        return newList;
    }
};

//Event監聽器。類似Unity的UnityEvent。可以註冊靜態或者類實例函數。支持刪除，不會重複。
template <typename T>
class Event {
private:
    typedef function<void(T)> EventHandler;
    struct ins_method_handler {
		std::any instance;
        std::any method;
		EventHandler handler;
    };
    vector<ins_method_handler> ins_handlers; 
    unordered_set<EventHandler*> handlers;
    vector<EventHandler> temp_handlers;
public:
	//temp listener cant be remove individually
    void AddListener(EventHandler f) {
        temp_handlers.push_back(f);
    }
    //static func, register address
    void AddListener(EventHandler* f) {
        handlers.insert(f);
    }
	//register instance method
    template <class C>
    void AddListener(C* ins, void (C::* method)(T)) {
		//check if this method is already added
        for (auto i : ins_handlers) {
			if (i.method.type() == typeid(method)) {
				if (any_cast<C*>(i.instance) == ins) {
					//same instance && same method address = already added.
                    return;
                }
			}
        }
		ins_handlers.push_back({ ins, method, bind(method, ins, placeholders::_1) });
    }
    void Invoke(T t) {
		for (auto f : temp_handlers) {
			f(t);
		}
		for (auto f : handlers) {
			(*f)(t);
		}
		for (auto imh : ins_handlers) {
			imh.handler(t);
		}
    }
	void RemoveListener(EventHandler* f) {
		handlers.erase(f);
	}
    template <class C>
    void RemoveListener(C* ins, void (C::* method)(T)) {
		for (auto it = ins_handlers.begin(); it != ins_handlers.end(); it++) {
			if (it->method.type() == typeid(method)) {
				if (any_cast<C*>(it->instance) == ins) {
					ins_handlers.erase(it);
					return;
				}
			}
		}
    }
    void RemoveAllListener() {
		handlers.clear();
		ins_handlers.clear();
		temp_handlers.clear();
    }
};
template <>
class Event<void> {
private:
    typedef function<void()> EventHandler;
    struct ins_method_handler{
		std::any instance;
        std::any method;
        EventHandler handler;
    };
    vector<ins_method_handler> ins_handlers;
    unordered_set<EventHandler*> handlers;
    vector<EventHandler> temp_handlers;
public:
    //temp listener cant be remove individually
    void AddListener(EventHandler f) {
        temp_handlers.push_back(f);
    }
    //static func, register address
    void AddListener(EventHandler* f) {
        handlers.insert(f);
    }
    //register instance method
    template <class C>
    void AddListener(C* ins, void (C::* method)()) {
        //check if this method is already added
        for (auto i : ins_handlers) {
            if (i.method.type() == typeid(method)) {
                if (any_cast<C*>(i.instance) == ins) {
                    //same instance && same method address = already added.
                    return;
                }
            }
        }
        ins_handlers.push_back({ ins, method, bind(method, ins) });
    }
    void Invoke() {
        for (auto f : temp_handlers) {
            f();
        }
        for (auto f : handlers) {
            (*f)();
        }
        for (auto imh : ins_handlers) {
            imh.handler();
        }
    }
    void RemoveListener(EventHandler* f) {
        handlers.erase(f);
    }
    template <class C>
    void RemoveListener(C* ins, void (C::* method)()) {
        for (auto it = ins_handlers.begin(); it != ins_handlers.end(); it++) {
            if (it->method.type() == typeid(method)) {
                if (any_cast<C*>(it->instance) == ins) {
                    ins_handlers.erase(it);
                    return;
                }
            }
        }
    }
    void RemoveAllListener() {
        handlers.clear();
        ins_handlers.clear();
        temp_handlers.clear();
    }
};

//網格模型頂點
struct Vertex {
    vec3 position;
    vec2 uv;
    vec3 normal;
};

// int key, int action. For keyboard input callback
struct keyCallbackData {
    int key;
    int action;
};

//int button, int action. For mouse input callback
struct mouseCallbackData {
    int button;
    int action;
};

//double xoffset, double yoffset. For scroll input callback
struct scrollCallbackData {
    double xOffset;
    double yOffset;
};

//double x, double y. for curosr position input callback.
struct cursorPositionCallbackData {
    double x;
    double y;
};

//Camera View & Proj Matrix Data, for Camera UBO
struct CameraVPData {
	vec3 cameraPosition;
    mat4 viewMatrix;
    mat4 projectionMatrix;
};

//單個光源資料
struct SingleLightData {
    LightType lightType = POINT_LIGHT;

    vec3 direction = vec3(-1.0f, -1.0f, -1.0f); //for directional & sopt light
    vec3 position = vec3(0.0f, 0.0f, 0.0f); //for point & spot light
    float cutOff = 30.0f; //for spot light
    float outerCutOff = 45.0f; //for spot light

    float constant = 0.0f;
    float linear = 0.0f;
    float quadratic = 1.0f;

    Color lightColor = Color::White;
    float intensity = 1.0f;
};

//SSBO data struct for light.Include ambientPower, ambientColor, singleLightDataCount, allSingleLightData
struct LightData_SSBO {
    //環境光資料
    float ambientPower = 1.0f;
    Color ambientLight = Color::Black;
    //其他單一光源資料
    //其他單一光源資料
    unsigned int singleLightDataLength = 0;
    vector<SingleLightData> allSingleLightData;
};

//for shadow map
struct ShadowLightData_PointLight {
	mat4 lightProj;
	mat4 lightViews[6];
    GLuint frameBuffer_ID; //深度幀緩衝
    GLuint shadowMap_ID; //深度貼圖
    float farPlane=0.0; // for fragment shader
	vec3 lightPos=vec3(0.0,0.0,0.0); //for fragment shader
};
struct ShadowLightData_OtherLight {
    mat4 lightVP; 
    GLuint frameBuffer_ID; //深度幀緩衝
    GLuint shadowMap_ID; //深度貼圖
    LightType lightType;
    vec3 lightPos_OR_lightDir; //lightPos for spot light, lightDir for directional light
};
struct FBO_Texture {
    GLuint FBO_ID;
    GLuint shadowMapTexture_ID;
    bool occupied = false;
};

#pragma region For Render Binding

//save for binding, in order for render
struct currentTextureData {
    unsigned int slot;
	GLuint textureID;
    vec2 offset;
    string uniformName;
};
//save one variable data in a material, in order for render
struct currentMaterialVariableData {
    string uniformName;
	ShaderUniformVariableType variableType;
	std::any value;
};
struct currentMaterialData {
    GameShader* shader;
	vector<currentTextureData> allTextureData;
	vector<currentMaterialVariableData> allVariableData;
    Color emissionColor = Color::Black;
    float emissionIntensity=1.0f;
    bool receiveShadow = true;
};
struct currentMeshData {
	GameMesh* mesh;
    mat4 modelMatrix;
};

#pragma endregion

//每幀的渲染資料
struct RenderFrameData {
    
    //renders
    vector<pair<function<void()>,GameShader*>> opaqueRenderFuncs; //用vector,因為渲染次序不重要
    multimap<int, pair<function<void()>, GameShader*>> transparentRenderFuncs; //有可能兩個物件的距離相同（優先度相同），所以使用multimap
    list<pair<function<void()>, GameShader*>> uiRenderFuncs; //深度優先遍歷物件樹可以直接對UI排序，無需使用multimap.用list方便在前面推入，讀取時從前面開始（越往後的UI越優先）

    //light
    vector<SingleLightData> allSingleLights;
    Color ambientLight = Color::Black;
    float ambientPower = 1.0f;

    //shadow light
    vector<ShadowLightData_OtherLight> allShadowLights_DirectionalLight;
    vector<ShadowLightData_OtherLight> allShadowLights_SpotLight;
    vector<ShadowLightData_PointLight> allShadowLights_PointLight;
    vector<pair<currentMeshData, GameTexture*>> allMeshesCastingShadow; // mesh & diffuse textuer data, for shadow mapping.

    //cameras
    CameraVPData cameraVPData;
    Color cameraBackgroundColor = Color::Black;
};
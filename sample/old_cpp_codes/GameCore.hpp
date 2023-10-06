//遊戲主要組成部分,包括GameObject, GameAttributeBasic, GameAttribute，TransformAttribute 
//因為這幾個類互相引用，拆開會很麻煩，所以放在一起
#pragma once
#include "Dependencies/glew/glew.h"
#include "Dependencies/glm/gtc/matrix_transform.hpp"
#include "DataStructure.hpp"
#include "Engine.h"
#include <iostream>
#include <typeinfo>
#include <type_traits>

#pragma region GameAttributeBasic
class GameObject;
class GameAttributeBasic {
protected:
	bool enable = true;
	bool started = false;
	GameObject* gameObject = nullptr;
	GameAttributeBasic(GameObject* gameObj, bool enable) {
		this->gameObject = gameObj;
		this->enable = enable;
		Awake();
		if (enable) this->OnEnable();
		else this->OnDisable();
	}
public:
	virtual int getAttributeTypePriority() = 0;
	virtual string getAttributeTypeName() = 0;
	virtual const type_info* getAttributeType() = 0;
	virtual void Destroy() = 0;

	virtual void Awake() {}
	virtual void Start() {}
	virtual void FixedUpdate() {}
	virtual void Update() {}
	virtual void LateUpdate() {}
	virtual void OnDestroy() {}
	//剛添加到GameObject的時候，OnEnable()或者OnDisable()會被呼叫，視乎enable的初始狀態
	virtual void OnEnable() {}
	virtual void OnDisable() {}

	virtual void setEnable(bool set) {
		if (set != this->enable) {
			this->enable = set;
			if (set) this->OnEnable();
			else this->OnDisable();
		}
	}
	bool isEnable() { return enable; }
	GameObject* getGameObject() { return gameObject; }
	bool EqualToAttributeType(const type_info* attType) { return attType == getAttributeType(); }
	//Run 之前要檢查是否enable
	void RunAttribute(EngineRunStage stage) {
		if (!started) {
			started = true;
			Start();
		}
		switch (stage) {
		case FIXED_UPDATE:
			FixedUpdate();
			break;
		case UPDATE:
			Update();
			break;
		case LATE_UPDATE:
			LateUpdate();
			break;
		}
	}
};


#pragma endregion

#pragma region GameAttribute
//遊戲屬性。掛載在遊戲物件GameObject上運行。繼承此類來創建新屬性
template <class Att, char* const AttributeTypeName, int AttTypePriority = 0>
class GameAttribute : public GameAttributeBasic {
protected:
	static unordered_set<GameAttributeBasic*> allAttInstance; //所有此類型的attribute實例, 方便查找
	GameAttribute(GameObject* gameObj, bool enable) :GameAttributeBasic(gameObj, enable) {}
public:
	virtual void Destroy() override {
		allAttInstance.erase(this);
		delete this;
	}
	virtual const type_info* getAttributeType() override { return GetAttributeType(); }
	virtual int getAttributeTypePriority() override { return GetAttPriority(); }
	virtual string getAttributeTypeName() override { return GetTypeName(); }

	static unordered_set<GameAttributeBasic*> getAllAttInstance() {
		return allAttInstance;
	}
	static string GetTypeName() {
		return string(AttributeTypeName);
	}
	static int GetAttPriority() {
		return AttTypePriority;
	}
	static const type_info* GetAttributeType() {
		return &typeid(Att);
	};
};
template <class Att, char* const AttributeTypeName, int AttTypePriority> unordered_set<GameAttributeBasic*> GameAttribute<Att, AttributeTypeName, AttTypePriority>::allAttInstance;
#pragma endregion

#pragma region GameObject
class TransformAttribute;
class GameObject {
protected:
	bool active = true;
	string name = "";
	unordered_set<string> tags;
	GameObject* parent = nullptr;
	unique_list<GameObject*> children; //child has sibling order, thus need list.
	multimap<int, GameAttributeBasic*> attributes; //attribute has priority order, thus need multimap.
	TransformAttribute* _transform = nullptr;

	static unordered_map<string, unordered_set<GameObject*>*> GameObjectNameSearchMap;
	static unordered_map<string, unordered_set<GameObject*>*> GameObjectTagSearchMap;
	static unique_list<GameObject*> RootGameObjects; //最頂層（root）物件列表. has sibling order, thus need list,
public:
	GameObject(string name = "", bool active = true, GameObject* parent = nullptr) {
		if (!(GameObjectNameSearchMap.count(name))) {
			GameObjectNameSearchMap[name] = new unordered_set<GameObject*>();
		}
		GameObjectNameSearchMap[name]->insert(this);
		this->active = active;
		if (parent == nullptr) RootGameObjects.push_back(this);
		else {
			this->parent = parent;
			parent->children.push_back(this);
		}
	}

	bool isActive() {
		return active;
	}
	//設定物件是否啟用，會啟用或不啟用所有子物件，並呼叫所有（啟用中）屬性的OnEnable()/OnDisable()
	void SetActive(bool set) {
		if (set == active) return;
		for (auto child : children) {
			child->SetActive(set);
		}
		active = set;
		for (auto att : multimap(attributes)) {
			if (att.second->isEnable()) {
				if (!set) att.second->OnDisable();
				else att.second->OnEnable();
			}
		}
	}
	void SetName(string newName) {
		int pos = 0;
		GameObjectNameSearchMap[name]->erase(this);
		if (!(GameObjectNameSearchMap.count(newName))) GameObjectNameSearchMap[newName] = new unordered_set<GameObject*>();
		GameObjectNameSearchMap[newName]->insert(this);
		this->name = name;
	}
	string getName() {
		return name;
	}
	bool hasTag(string tag) {
		return (bool)tags.count(tag);
	}
	void AddTag(string tag) {
		if (hasTag(tag)) return;
		tags.insert(tag);
		if (!(GameObjectTagSearchMap.count(tag))) GameObjectTagSearchMap[tag] = new unordered_set<GameObject*>();
		GameObjectTagSearchMap[tag]->insert(this);
	}
	void RemoveTag(string tag) {
		if (!hasTag(tag)) return;
		tags.erase(tag);
		GameObjectTagSearchMap[tag]->erase(this);
	}
	void Destroy() {
		if (parent != nullptr) parent->children.remove(this);
		else RootGameObjects.remove(this);
		for (auto att : multimap(attributes)) att.second->Destroy();
		for (auto child : children.copy()) child->Destroy();
		delete this;
	}

	void SetParent(GameObject* newParent) {
		if (newParent == parent) return;
		if (parent != nullptr) parent->children.remove(this);
		else RootGameObjects.remove(this);
		parent = newParent;
		if (parent != nullptr) parent->children.push_back(this);
		else RootGameObjects.push_back(this);
	}
	void MoveToRootObjectList() {
		return SetParent(nullptr);
	}
	GameObject* GetParent() { return parent; }
	bool hasChild(GameObject* child) {
		return (bool)children.contains(child);
	}
	unique_list<GameObject*>* GetAllChildren() { return &children; }
	GameObject* GetChild(int index) {
		if (index >= children.size()) return nullptr;
		return children[index];
	}
	int ChildCount() { return children.size(); }
	void ChangeSiblingIndex(int index) {
		if (index >= children.size()) return;
		if (parent == nullptr) {
			if (index >= RootGameObjects.size()) return;
			RootGameObjects.remove(this);
			RootGameObjects.insertAt(this, index);
		}
		else {
			if (index >= parent->children.size()) return;
			parent->children.remove(this);
			parent->children.insertAt(this, index);
		}
	}
	int GetSiblingIndex() {
		if (parent == nullptr) {
			for (int i = 0; i < RootGameObjects.size(); i++) {
				if (RootGameObjects[i] == this) return i;
			}
		}
		else {
			for (int i = 0; i < parent->children.size(); i++) {
				if (parent->children[i] == this) return i;
			}
		}
		return -1;
	}

	//Attribute
	bool hasAttributeType(const type_info* attType) {
		for (auto att : attributes) if (att.second->EqualToAttributeType(attType)) return true;
		return false;
	}
	template <class Att, typename enable_if < is_convertible<Att*, GameAttributeBasic*>{}, int > ::type = 0 >
	bool hasAttributeType() {
		return hasAttributeType(&typeid(Att));
	}
	bool hasAttribute(GameAttributeBasic* att) {
		for (auto a : attributes) if (a.second == att) return true;
		return false;
	}
	GameAttributeBasic* getFirstAttributeType(const type_info* attType) {
		for (auto att : attributes) if (att.second->EqualToAttributeType(attType)) return att.second;
		return nullptr;
	}
	template <class Att, typename enable_if < is_convertible<Att*, GameAttributeBasic*>{}, int > ::type = 0 >
	Att * getFirstAttributeType() {
		auto ptr = getFirstAttributeType(&typeid(Att));
		if (ptr != nullptr) return(Att*)ptr;
		return nullptr;
	}
	vector<GameAttributeBasic*> getAllAttributeType(const type_info* attType) {
		vector<GameAttributeBasic*> tempV;
		for (auto att : attributes) if (att.second->EqualToAttributeType(attType)) tempV.push_back(att.second);
		return tempV;
	}
	template <class Att, typename enable_if < is_convertible<Att*, GameAttributeBasic*>{}, int > ::type = 0 >
	vector<Att*> getAllAttributeType() {
		vector<Att*> tempV;
		auto attType = &typeid(Att);
		for (auto att : attributes) if (att.second->EqualToAttributeType(attType)) tempV.push_back((Att*)att);
		return tempV;
	}
	void RemoveAttributeType(const type_info* attType) {
		for (auto att : multimap(attributes)) if (att.second->EqualToAttributeType(attType)) att.second->Destroy();
	}
	template <class Att, typename enable_if < is_convertible<Att*, GameAttributeBasic*>{}, int > ::type = 0 >
	void RemoveAttributeType() {
		return RemoveAttributeType(&typeid(Att));
	}
	void RemoveAttribute(GameAttributeBasic* att) {
		for (auto it = attributes.begin(); it != attributes.end(); it++) {
			if (it->second == att) {
				attributes.erase(it);
				break;
			}
		}
	}
	multimap<int, GameAttributeBasic*>* GetAllAttributes() { return &attributes; }
	
	// the only way to add attribute and create a new attribute
	template <class Att, typename enable_if < is_convertible<Att*, GameAttributeBasic*>{}, int > ::type = 0, typename ...Args >
	Att* AddAttribute(bool enable = true, Args... args) {
		if (typeid(Att) == typeid(TransformAttribute)) return (Att*)transform();
		Att* newatt = new Att(this, enable, args...);
		attributes.insert(pair<int, GameAttributeBasic*>{ ((GameAttributeBasic*)newatt)->getAttributeTypePriority(), (GameAttributeBasic*)newatt});
		return newatt;
	}
	bool SetAllAttributeStatus(bool set) {
		for (auto att : attributes) {
			att.second->setEnable(set);
		}
	}
	
	TransformAttribute* transform();

	//static
	static unordered_set<GameObject*>* FindGameObjectsWithName(string name) {
		if (GameObjectNameSearchMap.count(name)) return GameObjectNameSearchMap[name];
		return nullptr;
	}
	static unordered_set<GameObject*>* FindGameObjectsWithTag(std::string tag) {
		if (GameObjectTagSearchMap.count(tag)) return GameObjectTagSearchMap[tag];
		return nullptr;
	}
	static unique_list<GameObject*>* GetRootGameObjects() { return &RootGameObjects; }
	static void ClearAllGameObject() {
		for (auto obj : RootGameObjects.copy()) obj->Destroy();
	}
	static void RunGameObject(GameObject* obj, EngineRunStage stage) {
		if (obj->isActive()) {
			for (auto att : obj->attributes) if (att.second->isEnable()) att.second->RunAttribute(stage);
			for (auto child : obj->children) RunGameObject(child, stage); //深度優先
		}
	}
};
unordered_map<string, unordered_set<GameObject*>*> GameObject::GameObjectNameSearchMap;
unordered_map<string, unordered_set<GameObject*>*> GameObject::GameObjectTagSearchMap;
unique_list<GameObject*> GameObject::RootGameObjects;

#pragma endregion

#pragma region TransformAttribute
char TransformAttributeName[] = "TransformAttribute";
/// <summary>
/// 類似Unity的Transform. 記載物件的位置、旋轉、縮放等資訊
/// </summary>
class TransformAttribute :public GameAttribute<TransformAttribute, TransformAttributeName> {
private:
	vec3 rotation = vec3(0, 0, 0);
	vec3 position = vec3(0, 0, 0);
	vec3 scale = vec3(1, 1, 1);
public:
	// TransformAttribute 一定是啟用狀態
	TransformAttribute(GameObject* gameObject, bool enable) :GameAttribute(gameObject, true) {}
	void setEnable(bool enable) override {
		cout << "TransformAttribute can't change enable status." << endl;
	}
	TransformAttribute* GetParentTransform() {
		GameObject* par = this->gameObject->GetParent();
		if (par == nullptr) return nullptr;
		return (par->transform());
	}
	void TidyUpRotation() {
		rotation.x = fmod(rotation.x, 360);
		rotation.y = fmod(rotation.y, 360);
		rotation.z = fmod(rotation.z, 360);
		if (rotation.x < 0) rotation.x = 360 + rotation.x;
		if (rotation.y < 0) rotation.y = 360 + rotation.y;
		if (rotation.z < 0) rotation.z = 360 + rotation.z;
	}
	vec3 GetLocalRotation() {
		TidyUpRotation();
		return rotation;
	}
	void SetLocalRotation(float x = NULL, float y = NULL, float z = NULL) {
		if (x != NULL) rotation.x += x;
		if (y != NULL) rotation.y += y;
		if (z != NULL) rotation.z += z;
		TidyUpRotation();
	}
	vec3 GetWorldPos() {
		auto parTransform = GetParentTransform();
		if (parTransform == nullptr) return position;
		return parTransform->GetWorldPos() + position;
	}
	vec3 GetWorldRotation() {
		auto parTransform = GetParentTransform();
		if (parTransform == nullptr) return rotation;
		vec3 rot = parTransform->GetWorldRotation() + rotation;
		rot.x = fmod(rot.x, 360);
		rot.y = fmod(rot.y, 360);
		rot.z = fmod(rot.z, 360);
		if (rot.x < 0) rot.x = 360 + rot.x;
		if (rot.y < 0) rot.y = 360 + rot.y;
		if (rot.z < 0) rot.z = 360 + rot.z;
		return rot;
	}
	vec3 GetWorldScale() {
		auto parTransform = GetParentTransform();
		if (parTransform == nullptr) return scale;
		return parTransform->GetWorldScale() * scale;
	}
	vec3 TransformDir(vec3 dir) {
		vec3 realRotation = GetWorldRotation();
		mat4 rotationMatrix = mat4(1.0);
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.x), vec3(1.0, 0.0, 0.0));
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.y), vec3(0.0, 1.0, 0.0));
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.z), vec3(0.0, 0.0, 1.0));
		return vec3(rotationMatrix * vec4(dir, 1.0));
	}
	vec3 TransformPoint(vec3 point) {
		vec3 realPos = GetWorldPos();
		return realPos + point;
	}
	vec3 InverseTransformPoint(vec3 point) {
		vec3 realPos = GetWorldPos();
		return point - realPos;
	}
	vec3 InverseTransformDir(vec3 dir) {
		vec3 realRotation = GetWorldRotation();
		mat4 rotationMatrix = mat4(1.0);
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.x), vec3(1.0, 0.0, 0.0));
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.y), vec3(0.0, 1.0, 0.0));
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.z), vec3(0.0, 0.0, 1.0));
		return vec3(inverse(rotationMatrix) * vec4(dir, 1.0));
	}
	void SetWorldPos(vec3 targetPos) {
		auto parTransform = GetParentTransform();
		if (parTransform == nullptr) {
			this->position = targetPos;
		}
		else {
			vec3 realPos = parTransform->GetWorldPos();
			position = targetPos - realPos;
		}
	}
	void SetWorldRotation(vec3 targetRotation) {
		auto parTransform = GetParentTransform();
		if (parTransform == nullptr) {
			rotation = targetRotation;
		}
		else {
			vec3 realRotation = parTransform->GetWorldRotation();
			rotation = targetRotation - realRotation;
			TidyUpRotation();
		}
	}
	void SetWorldScale(vec3 targetScale) {
		auto parTransform = GetParentTransform();
		if (parTransform == nullptr) {
			scale = targetScale;
		}
		else {
			vec3 realScale = parTransform->GetWorldScale();
			scale = targetScale / realScale;
		}
	}
	void SetForward(vec3 forwardDir) {
		vec3 realRotation = GetWorldRotation();
		mat4 rotationMatrix = glm::mat4(1.0);
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.x), vec3(1.0, 0.0, 0.0));
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.y), vec3(0.0, 1.0, 0.0));
		rotationMatrix = rotate(rotationMatrix, radians(realRotation.z), vec3(0.0, 0.0, 1.0));
		vec3 right = vec3(rotationMatrix * vec4(1.0, 0.0, 0.0, 1.0));
		vec3 up = vec3(rotationMatrix * vec4(0.0, 1.0, 0.0, 1.0));
		vec3 forward = vec3(rotationMatrix * vec4(0.0, 0.0, 1.0, 1.0));
		vec3 newForward = normalize(forwardDir);
		vec3 newRight = normalize(cross(up, newForward));
		vec3 newUp = normalize(cross(newForward, newRight));
		vec3 newRotation = vec3(0.0);
		newRotation.x = degrees(asin(newUp.z));
		newRotation.y = degrees(atan(newUp.x / newUp.y));
		newRotation.z = degrees(atan(newRight.z / newForward.z));
		SetWorldRotation(newRotation);
	}
	void LookAt(vec3 position) {
		vec3 realPos = GetWorldPos();
		vec3 forward = position - realPos;
		SetForward(forward);
	}
	void TransformObj(TransformAttribute* transform) {
		transform->SetWorldPos(TransformPoint(transform->GetWorldPos()));
		transform->SetWorldRotation(GetWorldRotation() + transform->GetWorldRotation());
	}
	vec3 forward() {
		return TransformDir(vec3(0.0, 0.0, 1.0));
	}
	void Rotate(vec3 rotateAxis, float rotateAngle) {
		mat4 rotationMatrix = mat4(1.0);
		rotationMatrix = rotate(rotationMatrix, radians(rotateAngle), rotateAxis);
		position = rotationMatrix * vec4(position, 1.0);
		SetForward(rotationMatrix * vec4(forward(), 1.0));
	}
	vec3 backward() {
		return -forward();
	}
	vec3 right() {
		return TransformDir(vec3(1.0, 0.0, 0.0));
	}
	void SetRight(vec3 rightDir) {
		vec3 newForward = normalize(cross(rightDir, right()));
		SetForward(newForward);
	}
	vec3 left() {
		return -right();
	}
	vec3 up() {
		return TransformDir(vec3(0.0, 1.0, 0.0));
	}
	void SetUp(vec3 upDir) {
		vec3 newRight = normalize(cross(upDir, up()));
		SetRight(newRight);
	}
	vec3 down() {
		return -up();
	}
	mat4 GetModelMatrix() {
		vec3 realPos = GetWorldPos();
		vec3 realRotation = GetWorldRotation();
		vec3 realScale = GetWorldScale();
		mat4 modelMatrix = mat4(1.0);
		modelMatrix = translate(modelMatrix, realPos);
		modelMatrix = rotate(modelMatrix, radians(realRotation.x), vec3(1.0, 0.0, 0.0));
		modelMatrix = rotate(modelMatrix, radians(realRotation.y), vec3(0.0, 1.0, 0.0));
		modelMatrix = rotate(modelMatrix, radians(realRotation.z), vec3(0.0, 0.0, 1.0));
		modelMatrix = glm::scale(modelMatrix, realScale);
		return modelMatrix;
	}
};

TransformAttribute* GameObject::transform() {
	if (_transform == nullptr) _transform = new TransformAttribute(this, true);
	return _transform;
}

#pragma endregion


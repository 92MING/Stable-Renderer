//包括GameShader, GameTexture，GameMaterial，GameMesh
#pragma once
#define STB_IMAGE_IMPLEMENTATION
#include "Dependencies/glew/glew.h"
#include "Dependencies/glm/gtc/type_ptr.hpp"
#include "Dependencies/glm/gtx/string_cast.hpp"
#include "Dependencies/stb_image/stb_image.h"
#include "DataStructure.hpp"
#include "Engine.h"
#include <fstream>
#include <iostream>


#pragma region GameShader

class GameShader {
private:
	static unordered_map<string, GLuint> shaderCache; //<路徑，ID>
	static unordered_map<string, int> shaderUseCount; // <路徑，使用數量>
	static unordered_map<string, GameShader*> AllShaders; //所有shader

	string name, VSPath, FSPath;
	GLuint ID = 0;
	GLuint vsID, fsID;

	string readShaderCode(const char* fileName) const
	{
		std::ifstream myInput(fileName);
		if (!myInput.good())
		{
			std::cout << "File failed to load..." << fileName << std::endl;
			exit(1);
		}
		return std::string(
			std::istreambuf_iterator<char>(myInput),
			std::istreambuf_iterator<char>()
		);
	}
	bool setupShader(const char* vertexPath, const char* fragmentPath) {
		cout << "setting up shader:" << name << endl;
		ID = glCreateProgram();
		std::string temp;

		if (shaderCache.count(vertexPath)) {
			glAttachShader(ID, shaderCache[vertexPath]);
			shaderUseCount[vertexPath]++;
		}
		else {
			vsID = glCreateShader(GL_VERTEX_SHADER);
			const GLchar* vCode;
			temp = readShaderCode(vertexPath);
			vCode = temp.c_str();
			glShaderSource(vsID, 1, &vCode, NULL);
			glCompileShader(vsID);
			if (!checkShaderStatus(vsID)) {
				cout << string(vertexPath) + " compile error" << endl;
				return false;
			}glAttachShader(ID, vsID);
			shaderCache[vertexPath] = vsID;
			shaderUseCount[vertexPath] = 1;
		}
		if (shaderCache.count(fragmentPath)) {
			glAttachShader(ID, shaderCache[fragmentPath]);
			shaderUseCount[fragmentPath]++;
		}
		else {
			fsID = glCreateShader(GL_FRAGMENT_SHADER);
			const GLchar* fCode;
			temp = readShaderCode(fragmentPath);
			fCode = temp.c_str();
			glShaderSource(fsID, 1, &fCode, NULL);
			glCompileShader(fsID);
			if (!checkShaderStatus(fsID)) {
				cout << string(fragmentPath) + " compile error" << endl;
				return false;
			}
			glAttachShader(ID, fsID);
			shaderCache[fragmentPath] = fsID;
			shaderUseCount[fragmentPath] = 1;
		}
		glLinkProgram(ID);
		if (!checkProgramStatus(ID)){
			cout << string(name) + " link error" << endl;
			return false;
		}
		//glDeleteShader(vertexShaderID);
		//glDeleteShader(fragmentShaderID);
		glUseProgram(0);
		cout << "has set up shader: " << name << endl;
		cout << "id = " << ID << endl;
		return true;
	}
	GameShader(string name, string VSPath, string FSPath) {
		this->name = name;
		if (setupShader(VSPath.c_str(), FSPath.c_str())) {
			this->VSPath = VSPath;
			this->FSPath = FSPath;
			AllShaders[name] = this;
		}
		else {
			Delete();
		}
	}

public:
	void use() const {
		glUseProgram(ID);
		cout << "using shader:" << name << endl;
	};
	GLuint getProgramID() const { return ID; }
	pair<string, string> getShaderPath() const {
		return pair<string, string>(VSPath, FSPath);
	}
	pair<GLuint, GLuint> getShaderID() const {
		return pair<GLuint, GLuint>(vsID, fsID);
	}
	string GetName()const { return name; }
	void setMat4(const string& name, mat4& value) {
		unsigned int transformLoc = glGetUniformLocation(ID, name.c_str());
		glUniformMatrix4fv(transformLoc, 1, GL_FALSE, glm::value_ptr(value));
	};
	void setMat3(const string& name, mat3& value) {
		unsigned int transformLoc = glGetUniformLocation(ID, name.c_str());
		glUniformMatrix3fv(transformLoc, 1, GL_FALSE, glm::value_ptr(value));
	};
	void setVec4(const string& name, float v1, float v2, float v3, float v4) {
		unsigned int transformLoc = glGetUniformLocation(ID, name.c_str());
		glUniform4f(transformLoc, v1, v2, v3, v4);
	};
	void setVec4(const string& name, vec4 value) {
		glUniform4fv(glGetUniformLocation(this->ID, name.c_str()), 1, &value[0]);
	}
	void setVec3(const string& name, vec3 value) const {
		glUniform3fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
	};
	void setVec3(const string& name, float v1, float v2, float v3) const {
		glUniform3f(glGetUniformLocation(ID, name.c_str()), v1, v2, v3);
	};
	void setVec2(const string& name, float v1, float v2)const {
		glUniform2f(glGetUniformLocation(ID, name.c_str()), v1, v2);
	}
	void setVec2(const string& name, vec2 value)const {
		glUniform2fv(glGetUniformLocation(ID, name.c_str()), 1, &value[0]);
	}
	void setFloat(const string& name, float value) const {
		glUniform1f(glGetUniformLocation(ID, name.c_str()), value);
	};
	void setInt(const string& name, int value) const {
		glUniform1i(glGetUniformLocation(ID, name.c_str()), value);
	};
	bool checkShaderStatus(GLuint shaderID) const
	{
		return checkStatus(shaderID, glGetShaderiv, glGetShaderInfoLog, GL_COMPILE_STATUS);
	}
	bool checkProgramStatus(GLuint programID) const {
		return checkStatus(programID, glGetProgramiv, glGetProgramInfoLog, GL_LINK_STATUS);
	}
	bool checkStatus(GLuint objectID, PFNGLGETSHADERIVPROC objectPropertyGetterFunc, PFNGLGETSHADERINFOLOGPROC getInfoLogFunc, GLenum statusType) const
	{
		GLint status;
		objectPropertyGetterFunc(objectID, statusType, &status);
		if (status != GL_TRUE)
		{
			GLint infoLogLength;
			objectPropertyGetterFunc(objectID, GL_INFO_LOG_LENGTH, &infoLogLength);
			GLchar* buffer = new GLchar[infoLogLength];

			GLsizei bufferSize;
			getInfoLogFunc(objectID, infoLogLength, &bufferSize, buffer);
			std::cout << buffer << std::endl;

			delete[] buffer;
			return false;
		}
		return true;
	}
	void Delete() {
		AllShaders.erase(name);
		shaderUseCount[VSPath]--;
		if (shaderUseCount[VSPath] == 0) {
			shaderCache.erase(VSPath);
			shaderUseCount.erase(VSPath);
			glDeleteShader(vsID);
		}
		shaderUseCount[FSPath]--;
		if (shaderUseCount[FSPath] == 0) {
			shaderCache.erase(FSPath);
			shaderUseCount.erase(FSPath);
			glDeleteShader(fsID);
		}
		delete this;
	}
	static GameShader* CreateShader(string name, string VSPath, string FSPath) {
		if (AllShaders.count(name)) {
			cout << "already exist Shader Name: " << name << endl;
			return nullptr;
		}
		GameShader* newShader = new GameShader(name, VSPath, FSPath);
		return newShader;
	}
	static GameShader* FindShader(string name) {
		if (AllShaders.count(name)) return AllShaders[name];
		cout << "Shader name : " << name << " isnt found" << endl;
		return nullptr;
	}
	static void DeleteShader(string name) {
		if (!AllShaders.count(name)) {
			cout << "Shader name : " << name << " isnt found" << endl;
			return;
		}
		FindShader(name)->Delete();
	}
	static string GetShaderVariableTypeString(ShaderUniformVariableType type) {
		switch (type) {
		case Int:
			return "int";
		case Float:
			return "float";
		case Vec2:
			return "vec2";
		case Vec3:
			return "vec3";
		case Vec4:
			return "vec4";
		case Mat3:
			return "mat3";
		case Mat4:
			return "mat4";
		default:
			return "";
		}
	}
};
unordered_map<string, unsigned int> GameShader::shaderCache;
unordered_map<string, int> GameShader::shaderUseCount;
unordered_map<string, GameShader*> GameShader::AllShaders;
#pragma endregion

#pragma region GameTexture

/// <summary>
/// 遊戲貼圖。生成後不可更改。
/// </summary>
class GameTexture {
private:
	static unordered_map<string, GameTexture*> AllTextures;
	static vec4 DefaultBoarderColor;

	GLuint ID = 0;
	int Width = 0, Height = 0;
	GLuint internalColorFormat;//紋理中顏色格式，例如rgba
	GLuint colorFormat; //顏色格式，例如rgba
	GLuint dataType;//資料類型, 例如float，unsigned byte
	TextureWrapMode mode;
	string name; //for search texture in resource
	vec4 boarderColor;

	void SetupTexture(const char* texturePath, TextureWrapMode mode = Repeat, vec4 boarderColor = DefaultBoarderColor)
	{
		stbi_set_flip_vertically_on_load(true);
		int BPP;
		unsigned char* data = stbi_load(texturePath, &Width, &Height, &BPP, 0);
		GLenum format = 3;
		switch (BPP) {
		case 1: format = GL_RED; break;
		case 3: format = GL_RGB; break;
		case 4: format = GL_RGBA; break;
		}
		this->internalColorFormat = format;
		this->colorFormat = format;

		glGenTextures(1, &ID);
		glBindTexture(GL_TEXTURE_2D, ID);

		this->mode = mode;
		switch (mode) {
		case Repeat:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			break;
		case Mirror_Repeat:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
			break;
		case Clamp_Edge:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			break;
		case Clamp_Boarder:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, &boarderColor[0]);
			break;
		}

		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR_MIPMAP_LINEAR);

		if (data) {
			glTexImage2D(GL_TEXTURE_2D, 0, format, Width, Height, 0, format, GL_UNSIGNED_BYTE, data);
			this->dataType = GL_UNSIGNED_BYTE;
			glGenerateMipmap(GL_TEXTURE_2D);
			stbi_image_free(data);
		}
		else {
			std::cout << "Failed to load texture: " << texturePath << std::endl;
			exit(1);
		}

		std::cout << "Load " << texturePath << " successfully!" << std::endl;
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	GameTexture(string name, string texturePath, TextureWrapMode mode, vec4 boarderColor = DefaultBoarderColor) {
		SetupTexture(texturePath.c_str(), mode, boarderColor);
		AllTextures[name] = this;
		this->name = name;
		this->boarderColor = boarderColor;
	}
	GameTexture(string name, int width, int height, TextureWrapMode mode = Clamp_Edge, GLuint internalColorFormat = GL_RGB16F, GLuint colorFormat = GL_RGB, GLuint dataType = GL_FLOAT, vec4 boarderColor = DefaultBoarderColor) {
		this->boarderColor = boarderColor;
		this->internalColorFormat = internalColorFormat;
		this->colorFormat = colorFormat;
		this->dataType = dataType;
		this->mode = mode;
		glGenTextures(1, &ID);
		glBindTexture(GL_TEXTURE_2D, ID);
		glTexImage2D(GL_TEXTURE_2D, 0, internalColorFormat, width, height, 0, colorFormat, dataType, NULL);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
		glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
		switch (mode) {
		case Repeat:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT);
			break;
		case Mirror_Repeat:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_MIRRORED_REPEAT);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_MIRRORED_REPEAT);
			break;
		case Clamp_Edge:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
			break;
		case Clamp_Boarder:
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_BORDER);
			glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_BORDER);
			glTexParameterfv(GL_TEXTURE_2D, GL_TEXTURE_BORDER_COLOR, &boarderColor[0]);
			break;
		}
		glBindTexture(GL_TEXTURE_2D, 0);
		AllTextures[name] = this;
		this->name = name;
	}
public:
	vec2 offset = vec2(0.0, 0.0); //貼圖偏移量

	void Delete() {
		AllTextures.erase(name);
		glDeleteTextures(1, &ID);
		delete this;
	}
	pair<int, int> GetWidthAndHeight() {
		return pair<int, int>(Width, Height);
	}
	GLuint GetTexID() const { return ID; }
	GLuint GetColorFormat() const {
		return colorFormat;
	}
	GLuint GetInternalColorFormat() const {
		return internalColorFormat;
	}
	GLuint GetDataType() const {
		return dataType;
	}
	string GetName() const {
		return name;
	}

	static GameTexture* CreatePicTexture(string name, string texturePath, TextureWrapMode mode = Repeat, vec4 boarderColor = DefaultBoarderColor) {
		if (AllTextures.count(name)) {
			cout << "Texture Name: " << name << " already exist" << endl;
			return nullptr;
		}
		return new GameTexture(name, texturePath, mode, boarderColor);
	}
	static GameTexture* CreateEmptyTexture(string name, int width, int height, TextureWrapMode mode = Clamp_Edge, GLuint internalColorFormat = GL_RGB16F, GLuint colorFormat = GL_FLOAT, GLuint dataType = GL_FLOAT, vec4 boarderColor = DefaultBoarderColor) {
		if (AllTextures.count(name)) {
			cout << "Texture Name: " << name << " already exist" << endl;
			return nullptr;
		}
		return new GameTexture(name, width, height, mode, internalColorFormat, colorFormat, dataType, boarderColor);
	}
	static GameTexture* FindTexture(string name) {
		if (!AllTextures.count(name)) {
			cout << "No Textuer: " << name << endl;
			return nullptr;
		}
		return AllTextures[name];
	}
	static void DeleteTexture(string name) {
		GameTexture* tex = FindTexture(name);
		if (tex == nullptr) return;
		tex->Delete();
	}
	static void BindTexture(unsigned int slot, GLuint texID, vec2 offset, string uniformName, GameShader &shader) {
		//cout << "binding tex || slot=" << slot << " texID=" << texID << " uniformName=" << uniformName << endl;
		glActiveTexture(GL_TEXTURE0 + Engine::GetCurrentTextureBindingPosition() + slot);
		glBindTexture(GL_TEXTURE_2D, texID);
		shader.setInt(uniformName, slot);
		glBindTexture(GL_TEXTURE_2D, 0);
	}
	static void SetDefailtBoarderColor(vec4 color) {
		DefaultBoarderColor = color;
	}
};
unordered_map<string, GameTexture*> GameTexture::AllTextures;
vec4 GameTexture::DefaultBoarderColor = vec4(0.0f, 0.0f, 0.0f, 0.0f);
#pragma endregion

#pragma region GameMaterial
class GameMaterial {
private:
	static unordered_map<string, GameMaterial*> AllMaterials;
	unordered_map<string, pair<ShaderUniformVariableType, void*>> shaderVariables;
	unique_list<pair<string, GameTexture*>> textures;
	string name;
	GameShader* shader = nullptr;
	GameMaterial(string name, RenderType renderType = Opaque, int renderOrder = 0, GameShader* shader = nullptr) {
		this->name = name;
		this->renderType = renderType;
		this->renderOrder = renderOrder;
		this->shader = shader;
	}
public:
	//common parameters
	RenderType renderType = Opaque;
	int renderOrder = 0;
	vec3 emissionColor = vec3(0.0f, 0.0f, 0.0f);
	float emissionIntensity = 1.0f;
	bool castShadow = true;
	bool receiveShadow = true;
	
	//shader
	void SetShader(string shaderName) {
		GameShader* shader = GameShader::FindShader(shaderName);
		if (shader != nullptr) this->shader = shader;
	}
	void SetShdaer(GameShader* shader) {
		this->shader = shader;
	}
	GameShader* GetShader() { return shader; }
	string GetShaderName() { return shader->GetName(); }
	
	//edit texture
	void PrintCurrentTextures() {
		int i = 0;
		cout << "(Material:" << name << ")Current textures:" << endl;
		for (auto name_tex : textures) {
			cout << "       Slot " << i << ":" << name_tex.first << " --> ";
			if (name_tex.second != nullptr) cout << "Texture: " << name_tex.second->GetName() << endl;
			else cout << "Empty" << endl;
			i++;
		}
	}
	bool checkTextureUniformName(string uniformName) {
		for (auto name_tex : textures) if (name_tex.first == uniformName) return true;
		return false;
	}
	void AddTextureSlot(string uniformName) {
		if (checkTextureUniformName(uniformName)) {
			cout << "("+name+")" << "Slot with uniform name: " << uniformName << "already exists" << endl;
			return;
		}
		string name_lower = uniformName;
		for (int i = 0; i < name_lower.size(); i++) name_lower[i] = tolower(name_lower[i]);
		textures.push_back({ name_lower, nullptr });
		cout << "Added texture slot:" + uniformName << endl;
	}
	void SetTextureSlot(string uniformName, GameTexture* texture) {
		string name_lower = uniformName;
		if (!checkTextureUniformName(uniformName)) {
			for (int i = 0; i < name_lower.size(); i++) name_lower[i] = tolower(name_lower[i]);
			textures.push_back({ name_lower, texture });
			cout << "(" + name + ")" << "Set texture slot:" + name_lower + " to texture:" + texture->GetName() << endl;
			return;
		}
		auto temp = textures.begin();
		for (int i = 0; i < textures.size(); i++) {
			if ((*temp).first == name_lower) {
				textures.setAt({ name_lower, texture }, i);
				return;
			}
			temp = next(temp, 1);
		}
		cout << "(" + name + ")" << "Set texture slot:" + name_lower + " to texture:" + texture->GetName() << endl;
	}
	void RemoveTextureSlot(string uniformName) {
		if (!checkTextureUniformName(uniformName)) {
			cout << "Texture slot: " << uniformName << " isn't found" << endl;
			return;
		}
		int i = 0;
		for (auto name_tex : textures) {
			if (name_tex.first == uniformName) break;
			i++;
		}
		textures.removeAt(i);
		cout << "Texture slot: " << uniformName << " deleted" << endl;
	}
	//do not care alphabet capitalization
	pair<int, GameTexture*> FindTextureSlot_byName(string uniformName) {
		string name_lower = uniformName;
		for (int i = 0; i < name_lower.size(); i++) name_lower[i] = tolower(name_lower[i]);
		if (!checkTextureUniformName(name_lower)) {
			cout << "(" + name + ")" << "Texture slot: " << name_lower << " isn't found" << endl;
			return { -1,nullptr };
		}
		int i = 0;
		for (auto name_tex : textures) {
			if ( name_tex.first == name_lower) break;
			i++;
		}
		return { i,textures[i].second };
	}
	pair<string, GameTexture*> FindTextureSlot_byIndex(int slot) {
		if (slot < 0 || slot >= textures.size()) {
			cout << "(" + name + ")" << "Texture slot: " << slot << " isn't found" << endl;
			return { "",nullptr };
		}
		return textures[slot];
	}
	void ChangeTextureSlotTo(string uniformName, unsigned int newSlot) {
		if (!checkTextureUniformName(uniformName)) {
			cout << "(" + name + ")" << "Texture slot name: " + uniformName + " isn't found" << endl;
			return;
		}
		if (newSlot >= textures.size()) {
			cout << "(" + name + ")" << "Texture slot: " << newSlot << " doesn't exist" << endl;
			return;
		}
		pair<int, GameTexture*> oldSlotData = FindTextureSlot_byName(uniformName);
		textures.removeAt(oldSlotData.first);
		textures.insertAt({ uniformName, oldSlotData.second }, newSlot);
		cout << "(" + name + ")" << "Texture slot: " << oldSlotData.first << " changed to slot: " << newSlot << endl;
	}
	void ChangeTextureSlotTo(unsigned int oldSlot, unsigned int newSlot) {
		if (oldSlot >= textures.size() || newSlot >= textures.size()) {
			cout << "(" + name + ")" << "Texture slot: " << oldSlot << " or " << newSlot << " doesn't exist" << endl;
			return;
		}
		pair<string, GameTexture*> oldSlotData = FindTextureSlot_byIndex(oldSlot);
		textures.removeAt(oldSlot);
		textures.insertAt(oldSlotData, newSlot);
		cout << "(" + name + ")" << "Texture slot: " << oldSlot << " changed to slot: " << newSlot << endl;
	}
	vector<currentTextureData> GetCurrentTextureData() {
		vector<currentTextureData> data;
		unsigned int i = 0;
		for (auto name_tex : textures) {
			auto uniformName = name_tex.first;
			auto tex = name_tex.second;
			data.push_back(currentTextureData{ i, tex->GetTexID(), tex->offset, uniformName});
			i++;
		}
		return data;
	}

	//edit shader variables
	void AddShaderVariable(string uniformName, ShaderUniformVariableType type) {
		if (shaderVariables.count(uniformName)) {
			cout << "(" + name + ")" << "shader variable:" + uniformName + " already exists" << endl;
			return;
		}
		shaderVariables[uniformName] = pair<ShaderUniformVariableType, void*>(type, nullptr);
	}
	void SetShaderVariable(string uniformName, ShaderUniformVariableType type, void* pointer) {
		if (!shaderVariables.count(uniformName)) AddShaderVariable(uniformName, type);
		shaderVariables[uniformName] = pair<ShaderUniformVariableType, void*>(type, pointer);
	}
	void RemoveShaderVariable(string uniformName) {
		if (!shaderVariables.count(uniformName)) {
			cout << "(" + name + ")" << "Shader variable name:" + uniformName + " isn't exists." << endl;
			return;
		}
		shaderVariables.erase(uniformName);
	}
	void PrintCurrentShaderVariables() {
		for (auto name_var : shaderVariables) {
			cout << name_var.first + "(" + GameShader::GetShaderVariableTypeString(name_var.second.first) + ")" + " : ";
			switch (name_var.second.first) {
			case Int:
				cout << *(int*)name_var.second.second << endl;
				break;
			case Float:
				cout << *(float*)name_var.second.second << endl;
				break;
			case Vec2:
				vec2 v2 = *(vec2*)name_var.second.second;
				cout << v2.x << " " << v2.y << endl;
				break;
			case Vec3:
				vec3 v3 = *(vec3*)name_var.second.second;
				cout << v3.x << " " << v3.y << " " << v3.z << endl;
				break;
			case Vec4:
				vec4 v4 = *(vec4*)name_var.second.second;
				cout << v4.x << " " << v4.y << " " << v4.z << " " << v4.w << endl;
				break;
			case Mat3:
				cout << endl;
				mat3 m3 = *(mat3*)name_var.second.second;
				cout << to_string(m3) << endl;
				break;
			case Mat4:
				cout << endl;
				mat4 m4 = *(mat4*)name_var.second.second;
				cout << to_string(m4) << endl;
				break;
			default:
				break;
			}
			
		}
	}
	unordered_map<string, pair<ShaderUniformVariableType, void*>> GetAllShaderVariables() const {
		return shaderVariables;
	}
	vector<currentMaterialVariableData> GetCurrentMaterialVariableData() {
		vector<currentMaterialVariableData> data;
		for (auto name_var : shaderVariables) {
			auto uniformName = name_var.first;
			auto type = name_var.second.first;
			std::any value;
			switch (type) {
				case Int:
					value = *(int*)name_var.second.second;
					break;
				case Float:
					value = *(float*)name_var.second.second;
					break;
				case Vec2:
					value = *(vec2*)name_var.second.second;
					break;
				case Vec3:
					value = *(vec3*)name_var.second.second;
					break;
				case Vec4:
					value = *(vec4*)name_var.second.second;
					break;
				case Mat3:
					value = *(mat3*)name_var.second.second;
					break;
				case Mat4:
					value = *(mat4*)name_var.second.second;
					break;
				default:
					value = NULL;
					break;
			}
			data.push_back(currentMaterialVariableData{ uniformName, type, value });
		}
		return data;
	}
	
	void Delete() {
		AllMaterials.erase(name);
		delete this;
	}
	
	static GameMaterial* CreateMaterial(string name, RenderType renderType = Opaque, int renderOrder = 0, GameShader* shader = nullptr) {
		if (AllMaterials.count(name)) {
			cout << "Material name: " << name << " already exists" << endl;
			return nullptr;
		}
		return new GameMaterial(name, renderType, renderOrder, shader);
	}
	static GameMaterial* FindMaterial(string name) {
		if (AllMaterials.count(name)) return AllMaterials[name];
		cout << "Material name: " << name << "isn't found" << endl;
		return nullptr;
	}
	static void DeleteMaterial(string name) {
		if (!AllMaterials.count(name)) {
			cout << "Material name: " << name << "isn't found" << endl;
			return;
		}
		AllMaterials[name]->Delete();
	}
};
unordered_map<string, GameMaterial*> GameMaterial::AllMaterials;
#pragma endregion

#pragma region GameMesh
const string SupportedSuffix[] = { "obj" };
class GameMesh {
private:
	static unordered_map<string, GameMesh*> AllMeshes_byName;
	static unordered_map<string, GameMesh*> AllMeshes_byPath;
	GLuint VAO, VBO, indexBufferID;
	string name;
	string path;
	vector<Vertex> vertices; //include pos,uv,normal
	vector<unsigned int> indices;
	void sendDataToOpenGL() {
		glGenVertexArrays(1, &VAO);
		glBindVertexArray(VAO);

		glGenBuffers(1, &VBO);
		glBindBuffer(GL_ARRAY_BUFFER,VBO);
		glBufferData(GL_ARRAY_BUFFER, vertices.size() * sizeof(Vertex), &vertices[0], GL_STATIC_DRAW);

		glGenBuffers(1, &indexBufferID);
		glBindBuffer(GL_ELEMENT_ARRAY_BUFFER, indexBufferID);
		glBufferData(GL_ELEMENT_ARRAY_BUFFER, indices.size() * sizeof(unsigned int), &indices[0], GL_STATIC_DRAW);
		
		glEnableVertexAttribArray(0);
		glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, position));

		glEnableVertexAttribArray(1);
		glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, uv));

		glEnableVertexAttribArray(2);
		glVertexAttribPointer(2, 3, GL_FLOAT, GL_FALSE, sizeof(Vertex), (void*)offsetof(Vertex, normal));
		
		cout << "sent model data to opengl. VAO:" << VAO << " ,name:" << name << endl;
	}
	void loadModel_OBJ(const char* objPath)
	{
		struct V {
			// struct for identify if a vertex has showed up
			unsigned int index_position, index_uv, index_normal;
			bool operator == (const V& v) const {
				return index_position == v.index_position && index_uv == v.index_uv && index_normal == v.index_normal;
			}
			bool operator < (const V& v) const {
				return (index_position < v.index_position) ||
					(index_position == v.index_position && index_uv < v.index_uv) ||
					(index_position == v.index_position && index_uv == v.index_uv && index_normal < v.index_normal);
			}
		};
		vector<vec3> temp_positions;
		vector<vec2> temp_uvs;
		vector<vec3> temp_normals;
		map<V, unsigned int> temp_vertices;

		unsigned int num_vertices = 0;
		cout << "\nLoading OBJ file " << objPath << "..." << endl;
		ifstream file;
		file.open(objPath);
		if (file.fail()) {
			cerr << "Impossible to open the file!" << endl;
			exit(1);
		}
		while (!file.eof()) {
			// process the object file
			char lineHeader[128];
			file >> lineHeader;

			if (strcmp(lineHeader, "v") == 0) {
				// geometric vertices
				vec3 position;
				file >> position.x >> position.y >> position.z;
				temp_positions.push_back(position);
			}
			else if (strcmp(lineHeader, "vt") == 0) {
				// texture coordinates
				vec2 uv;
				file >> uv.x >> uv.y;
				temp_uvs.push_back(uv);
			}
			else if (strcmp(lineHeader, "vn") == 0) {
				// vertex normals
				vec3 normal;
				file >> normal.x >> normal.y >> normal.z;
				temp_normals.push_back(normal);
			}
			else if (strcmp(lineHeader, "f") == 0) {
				// Face elements
				V vertices[3];
				for (int i = 0; i < 3; i++) {
					char ch;
					file >> vertices[i].index_position >> ch >> vertices[i].index_uv >> ch >> vertices[i].index_normal;
				}

				// Check if there are more than three vertices in one face.
				string redundency;
				getline(file, redundency);
				if (redundency.length() >= 5) {
					cerr << "There may exist some errors while load the obj file. Error content: [" << redundency << " ]" << endl;
					cerr << "Please note that we only support the faces drawing with triangles. There are more than three vertices in one face." << endl;
					cerr << "Your obj file can't be read properly by our simple parser :-( Try exporting with other options." << endl;
					exit(1);
				}
				for (int i = 0; i < 3; i++) {
					if (temp_vertices.find(vertices[i]) == temp_vertices.end()) {
						// the vertex never shows before
						Vertex vertex;
						vertex.position = temp_positions[vertices[i].index_position - 1];
						vertex.uv = temp_uvs[vertices[i].index_uv - 1];
						vertex.normal = temp_normals[vertices[i].index_normal - 1];

						this->vertices.push_back(vertex);
						this->indices.push_back(num_vertices);
						temp_vertices[vertices[i]] = num_vertices;
						num_vertices += 1;
					}
					else {
						// reuse the existing vertex
						unsigned int index = temp_vertices[vertices[i]];
						this->indices.push_back(index);
					}
				}
			}
			else {
				// it's not a vertex, texture coordinate, normal or face
				char stupidBuffer[1024];
				file.getline(stupidBuffer, 1024);
			}
		}
		file.close();
		cout << "Loaded mesh: "<<objPath << " .There are " << num_vertices << " vertices in the obj file.\n" << endl;
	}
	GameMesh(string name, string path) {
		this->name = name;
		this->path = path;
		AllMeshes_byName[name] = this;
		AllMeshes_byPath[path] = this;
		loadModel_OBJ(path.c_str());
		sendDataToOpenGL();
	}
public:
	void Draw() {
		//cout << "before draw:" << glGetError() << endl;
		glBindVertexArray(VAO);
		//cout << "after bind:" << glGetError() << endl;
		glDrawElements(GL_TRIANGLES, indices.size(), GL_UNSIGNED_INT, 0);
		//cout << "after draw:" << glGetError() << endl;
	}
	void Delete() {
		AllMeshes_byName.erase(name);
		AllMeshes_byPath.erase(path);
		glDeleteVertexArrays(1, &VAO);
		glDeleteBuffers(1, &VBO);
		glDeleteBuffers(1, &indexBufferID);
		delete this;
	}
	GLuint GetVAO() const {
		return VAO;
	}
	int GetIndiceCount() {
		return indices.size();
	}
	string GetName() const {
		return name;
	}
	string GetPath() const {
		return path;
	}
	vector<Vertex>* GetVertices() {
		return &vertices;
	}
	vector<unsigned int>* GetIndices() {
		return &indices;
	}
	int GetIndiceCount() const {
		return indices.size()/3;
	}
	/// <summary>
	/// 僅支持三角面的.OBJ
	/// </summary>
	/// <param name="name">自定義名稱</param>
	/// <param name="path">文件路徑</param>
	/// <returns></returns>
	static GameMesh* CreateMesh(string name, string path) {
		if (name == "" || path == "") {
			cout << "Name or path can't be empty" << endl;
			return nullptr;
		}
		if (AllMeshes_byName.count(name)) {
			cout << "Mesh(name): " << name << " already exist, path = '" << path << "'" << endl;
			return nullptr;
		}
		if (AllMeshes_byPath.count(path)) {
			cout << "Mesh(path): " << path << " already exist, name='" << name << "'" << endl;
			return nullptr;
		}
		if (!(path.find_last_of('.') == string::npos)) {
			string temp = path.substr(path.find_last_of('.') + 1, path.size() - 1);
			string suffix = "";
			for (auto c : temp) suffix += tolower(c);
			for (auto suf : SupportedSuffix) {
				if (suf == suffix) return new GameMesh(name, path);
			}
		}
		cout << "no supported files found. Supported file: ";
		for (auto s : SupportedSuffix) cout << " ." << s;
		cout << endl;
		return nullptr;
	}
	static GameMesh* FindMesh_ByName(string name) {
		if (AllMeshes_byName.count(name)) return AllMeshes_byName[name];
		cout << "Mesh(name):" << name << "isn't found" << endl;
		return nullptr;
	}
	static GameMesh* FindMesh_ByPath(string path) {
		if (AllMeshes_byPath.count(path)) return AllMeshes_byPath[path];
		cout << "Mesh(path):" << path << "isn't found" << endl;
		return nullptr;
	}
	static void DeleteMesh_ByName(string name) {
		if (!AllMeshes_byName.count(name)) {
			cout << "Mesh(name):" << name << "isn't found" << endl;
			return;
		}
		AllMeshes_byName[name]->Delete();
	}
	static void DeleteMesh_ByPath(string path) {
		if (!AllMeshes_byPath.count(path)) {
			cout << "Mesh(path):" << path << "isn't found" << endl;
			return;
		}
		AllMeshes_byPath[path]->Delete();
	}
};
unordered_map<string, GameMesh*> GameMesh::AllMeshes_byName;
unordered_map<string, GameMesh*> GameMesh::AllMeshes_byPath;
#pragma endregion

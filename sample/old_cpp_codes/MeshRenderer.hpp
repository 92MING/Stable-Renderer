#pragma once
#include "Renderer.hpp"

#pragma region MeshRenderer

class MeshRenderer : public RendererBasic {
protected:
	virtual mat4 GetModelMatrix() { return mat4(1.0f); }
public:
	GameMesh* mesh;
	GameMaterial* material;
	bool hasMesh() { return mesh != nullptr; }
	bool hasMaterial() { return material != nullptr; }
	
	virtual void SubmitTaskToEngine() override {
		if (mesh == nullptr || material==nullptr) return;
		
		currentMaterialData materialData;
		materialData.shader = material->GetShader();
		materialData.allTextureData = material->GetCurrentTextureData();
		materialData.allVariableData = material->GetCurrentMaterialVariableData();
		materialData.emissionColor = material->emissionColor;
		materialData.emissionIntensity = material->emissionIntensity;
		materialData.receiveShadow = material->receiveShadow;

		currentMeshData meshData;
		meshData.mesh = mesh;
		meshData.modelMatrix = GetModelMatrix();
		
		auto index_diffuseTex = material->FindTextureSlot_byName("diffuse");

		if (material->castShadow) Engine::AddCastingShadowMeshDataToRenderFrameData(meshData, index_diffuseTex.second);
		Engine::AddBindedRenderFuncToRenderFrameData(bind(MeshRenderer::Render, meshData, materialData), material->GetShader(), material->renderType, material->renderOrder);
	};
	static void Render(currentMeshData &meshData, currentMaterialData &materialData)
	{
		if (materialData.shader == nullptr) return; // shader is deleted in unknown reason
		materialData.shader->use();
		
		materialData.shader->setInt("receiveShadow", (int)materialData.receiveShadow);
		
		//set shader textures
		materialData.shader->setMat4("modelMatrix",meshData.modelMatrix);
		mat4 inverseTranspose_M = transpose(inverse(meshData.modelMatrix));
		materialData.shader->setMat4("modelMatrix_InverseTranspose", inverseTranspose_M);
		
		for (auto texData : materialData.allTextureData) 
			GameTexture::BindTexture(texData.slot, texData.textureID, texData.offset, texData.uniformName, *materialData.shader);
		
		// set shader uniform variables 
		for (auto variableData : materialData.allVariableData) {
			switch (variableData.variableType) {
				case Int:
					materialData.shader->setInt(variableData.uniformName, any_cast<int&>(variableData.value));
					break;
				case Float:
					materialData.shader->setFloat(variableData.uniformName, any_cast<float&>(variableData.value));
					break;
				case Vec2:
					materialData.shader->setVec2(variableData.uniformName, any_cast<vec2&>(variableData.value));
					break;
				case Vec3:
					materialData.shader->setVec3(variableData.uniformName, any_cast<vec3&>(variableData.value));
					break;
				case Vec4:
					materialData.shader->setVec4(variableData.uniformName, any_cast<vec4&>(variableData.value));
					break;
				case Mat3:
					materialData.shader->setMat3(variableData.uniformName, any_cast<mat3&>(variableData.value));
					break;
				case Mat4:
					materialData.shader->setMat4(variableData.uniformName, any_cast<mat4&>(variableData.value));
					break;
				default:
					break;
			}
		}
		materialData.shader->setVec3("emissionColor", materialData.emissionColor.rgb);
		materialData.shader->setFloat("emissionIntensity", materialData.emissionIntensity);
		
		meshData.mesh->Draw();
	}
};

#pragma endregion

#pragma region MeshRendererAttribute

char meshRendererAttName[] = "MeshRendererAttribute";
class MeshRendererAttribute :public GameAttribute<MeshRendererAttribute, meshRendererAttName>, public MeshRenderer {
protected:
	virtual mat4 GetModelMatrix() override {
		return gameObject->transform()->GetModelMatrix();
	}
public:
	MeshRendererAttribute(GameObject* gameObject, bool enable) :GameAttribute(gameObject, enable) {}
	
	void LateUpdate() override {
		if (mesh != nullptr && material != nullptr) {
			SubmitTaskToEngine();
		}
	}
};

#pragma endregion
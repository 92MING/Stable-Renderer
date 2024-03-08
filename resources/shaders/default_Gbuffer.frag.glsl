// default FS for late shading
#version 330 core
#define MAX_LIGHTS_NUM 256  // this constant will be edit by python script
layout (std140) uniform Matrices {
	mat4 model;
	mat4 view;
	mat4 projection;
	mat4 MVP;
	mat4 MVP_IT; // inverse transpose of MVP
	mat4 MV; // model-view matrix
	mat4 MV_IT; // inverse transpose of MV
	vec3 cameraPos;
	vec3 cameraDir;
};

// depth is output directly from z-buffer
layout (location = 0) out vec4 outColor; // rgba
layout (location = 1) out ivec4 outID;  // outID = (objID, material id, uv_Xcoord, uv_Ycoord)
layout (location = 2) out vec3 outPos; // global pos, output alpha for mask
layout (location = 3) out vec4 outNormal;	//normal (in view space),  output alpha for mask
layout (location = 4) out vec4 outNoise; // noise, in latent shape

// textures
uniform sampler2D diffuseTex;	// 0
uniform int hasDiffuseTex;
uniform sampler2D normalTex;	// 1
uniform int hasNormalTex;
uniform sampler2D specularTex;	// 2
uniform int hasSpecularTex;
uniform sampler2D emissionTex;	// 3
uniform int hasEmissionTex;
uniform sampler2D occlusionTex;	// 4
uniform int hasOcclusionTex;
uniform sampler2D metallicTex;	// 5
uniform int hasMetallicTex;
uniform sampler2D roughnessTex;	// 6
uniform int hasRoughnessTex;
uniform sampler2D displacementTex;	// 7
uniform int hasDisplacementTex;	
uniform sampler2D alphaTex;	// 8
uniform int hasAlphaTex;
uniform sampler2D noiseTex;	// 9 (currently only support 1 channel noise texture)
uniform int hasNoiseTex;

// material
uniform int objID;
uniform int materialID;

// from VS
in vec3 modelPos;
in vec3 worldPos;
in vec3 modelNormal;	// not normal from normal map!
in vec3 worldNormal;	// not normal from normal map!
in vec3 viewNormal;		// not normal from normal map!
in vec2 uv;
in vec3 modelTangent;

void main() {

	// get color
	if (hasDiffuseTex == 0)
		outColor = vec4(1.0, 0.0, 1.0, 1.0); // pink color means no texture
	else
		outColor = texture(diffuseTex, uv).rgba;
	
	// get noise
	if (hasNoiseTex == 0)
		outNoise = vec4(0.0, 0.0, 0.0, 0.0); // no noise texture
	else
		outNoise = texture(noiseTex, uv).rgba;

	// get position
    outPos = worldPos;

	// get normal
	if (hasNormalTex == 0){  // if no normal texture, use normal from mesh data
		outNormal = vec4(normalize(viewNormal) * 0.5 + 0.5, 1.0);
	}
	else{
		vec3 bitangent = cross(modelNormal, modelTangent);
		mat3 TBN = mat3(modelTangent, bitangent, modelNormal);
		vec3 real_model_normal = normalize(texture(normalTex, uv).rgb * 2.0 - 1.0);
		real_model_normal = normalize(TBN * real_model_normal);
		outNormal = vec4(texture(normalTex, uv).rgb, 1.0);
	}

	// get id
	ivec2 uvi = ivec2(uv * ivec2(textureSize(diffuseTex, 0)));
	outID = ivec4(objID, materialID, uvi.x, uvi.y);
}

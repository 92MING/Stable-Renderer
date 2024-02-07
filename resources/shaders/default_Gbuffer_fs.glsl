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

layout (location = 0) out vec4 outColorAndDepth; // (r, g, b, depth)
layout (location = 1) out vec3 outWorldPos; // global pos
layout (location = 2) out vec3 outTangentNormal;	//normal (in tangent space)
layout (location = 3) out vec3 outWorldNormal; // normal (in world space)
layout (location = 4) out ivec4 outID;  // outID = (objID, material id, uv_Xcoord, uv_Ycoord)

uniform sampler2D diffuseTex;
uniform int hasDiffuseTex;
uniform sampler2D normalTex;
uniform int hasNormalTex;

uniform int objID;
uniform int materialID;

in vec3 modelPos;
in vec3 worldPos;
in vec3 modelNormal;	// not normal from normal map!
in vec3 worldNormal;	// not normal from normal map!
in vec3 viewNormal;		// not normal from normal map!
in vec2 uv;
in vec3 modelTangent;

void main() {

	// get color & depth
	if (hasDiffuseTex == 0)
		outColorAndDepth = vec4(1.0, 0.0, 1.0, 1.0 - gl_FragCoord.z); // pink color means no texture
	else{
		vec3 outColor = texture(diffuseTex, uv).rgb;
		outColorAndDepth = vec4(outColor, 1.0 - gl_FragCoord.z);
	}

	// get position
    outWorldPos = worldPos;

	// get normal
	if (hasNormalTex == 0){  // if no normal texture, use normal from mesh data
		outTangentNormal = normalize(viewNormal) * 0.5 + 0.5;
		outWorldNormal = worldNormal;
	}
	else{
		vec3 bitangent = cross(modelNormal, modelTangent);
		mat3 TBN = mat3(modelTangent, bitangent, modelNormal);
		vec3 real_model_normal = normalize(texture(normalTex, uv).rgb * 2.0 - 1.0);
		real_model_normal = normalize(TBN * real_model_normal);
		outTangentNormal = texture(normalTex, uv).rgb;
		outWorldNormal = normalize((inverse(transpose(mat3(model))) * real_model_normal));
	}

	// get id
	ivec2 uv = ivec2(uv * ivec2(textureSize(diffuseTex, 0)));
	outID = ivec4(objID, materialID, uv.x, uv.y);
}

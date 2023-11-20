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
layout (location = 1) out vec3 outPos; // global pos
layout (location = 2) out vec3 outNormal; // normal (in view space)
layout (location = 3) out vec3 outWorldNormal; // normal (in world space)
layout (location = 4) out ivec4 outID;  // outID = (objID, material id, uv_Xcoord, uv_Ycoord)

uniform sampler2D diffuseTex;
uniform int hasDiffuseTex;
uniform sampler2D normalTex;
uniform int hasNormalTex;

uniform int objID;
uniform int materialID;

in vec3 worldPos;
in vec3 worldNormal; // normal in world space
in vec3 viewNormal; // normal in view space
in vec2 vertexUV;

void main() {

	// get color & depth
	if (hasDiffuseTex == 0)
		outColorAndDepth = vec4(1.0, 0.0, 1.0, 1.0 - gl_FragCoord.z); // pink color means no texture
	else{
		vec3 outColor = texture(diffuseTex, vertexUV).rgb;
		outColorAndDepth = vec4(outColor, 1.0 - gl_FragCoord.z);
	}

	// get position
    outPos = worldPos;

	// get normal
	if (hasNormalTex == 0){
		outNormal = viewNormal;
		outWorldNormal = worldNormal;
	}
	else{
		outNormal = normalize(MV_IT * vec4(texture(normalTex, vertexUV).rgb, 0.0)).xyz;
		outWorldNormal = normalize(inverse(transpose(model)) * vec4(texture(normalTex, vertexUV).rgb, 0.0)).xyz;
	}

	// get id
	ivec2 uv = ivec2(vertexUV * ivec2(textureSize(diffuseTex, 0)));
	outID = ivec4(objID, materialID, uv.x, uv.y);
}

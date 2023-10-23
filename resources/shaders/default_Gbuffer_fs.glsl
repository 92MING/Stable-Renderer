// default FS for late shading
#version 330 core
#define MAX_LIGHTS_NUM 256  // this constant will be edit by python script
layout (std140, binding=0) uniform Matrices {
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
layout (std140, binding=2) uniform CorrMap {
	int minimizeRatio;
};

layout (location = 0) out vec4 outColorAndDepth; // (r, g, b, depth)
layout (location = 1) out vec3 outPos;
layout (location = 2) out vec3 outNormal;
layout (location = 3) out ivec3 outID;  // outID = (objID, uv_Xcoord, uv_Ycoord)

uniform sampler2D diffuseTex;
uniform sampler2D normalTex;
uniform int objID;

in vec3 worldPos;
in vec3 worldNormal;
in vec2 vertexUV;

void main() {
	ivec2 texture_size = ivec2(textureSize(diffuseTex, 0));
	float minimizeSize = sqrt(float(minimizeRatio));
	ivec2 uv_int = ivec2(vertexUV * ivec2(textureSize(diffuseTex, 0)) / minimizeSize);
	// discard if uv_int is not the left-top corner of a block
	if (uv_int.x % minimizeRatio != 0 || uv_int.y % minimizeRatio != 0) {
		discard;
	}

	vec2 uv_float = vec2(uv_int) * minimizeSize / vec2(texture_size);

	// get color & depth
	vec3 outColor = vec3(0.0, 0.0, 0.0);
	for (int i = 0; i < int(minimizeSize); i++) {
		for (int j = 0; j < int(minimizeSize); j++) {
			outColor += texture(diffuseTex, uv_float + vec2(i, j) / vec2(texture_size)).rgb;
		}
	}
	outColor /= float(minimizeRatio);
	outColorAndDepth = vec4(outColor, 1.0 - gl_FragCoord.z);

	// get position
    outPos = worldPos;

	// get normal
    outNormal = normalize((MV_IT * vec4(texture(normalTex, vertexUV).xyz, 0.0)).xyz);

	// get id

	outID = ivec3(objID, uv_int.x, uv_int.y);
}

#version 430
#define MAX_SHADOW_LIGHT_NUM 8
#define MAX_LIGHT_NUM 64
#define DIRECTIONAL_LIGHT 0
#define SPOT_LIGHT 1
#define POINT_LIGHT 2

//camera data
layout (std140, binding = 0) uniform Camera {
	vec3 camPos;
    mat4 camViewMatrix;
    mat4 camProjMatrix;
};

//data of mesh
in layout(location=0) vec3 vertexPos;
in layout(location=1) vec2 vertexUV;
in layout(location=2) vec3 vertexNormal;

//directional light shadow map
uniform sampler2DArray shadowMapTexture2DArray_DirectionalLight;
uniform vec3 DirectionalLightDirections[MAX_SHADOW_LIGHT_NUM];
uniform mat4 DirectLightVP_Bias[MAX_SHADOW_LIGHT_NUM];
uniform int DirectionalLightShadowCount;

//spot light shadow map
uniform sampler2DArray shadowMapTexture2DArray_SpotLight;
uniform vec3 SpotLightPositions[MAX_SHADOW_LIGHT_NUM];
uniform mat4 SpotLightVP_Bias[MAX_SHADOW_LIGHT_NUM];
uniform int SpotLightShadowCount;

//point light shadow map
uniform samplerCubeArray shadowMapTextureCubeArray_PointLight;
uniform vec3 PointLightPositions[MAX_SHADOW_LIGHT_NUM];
uniform int PointLightShadowCount;
uniform float PointLightFarPlanes[MAX_SHADOW_LIGHT_NUM];

//texture
uniform sampler2D normal;

//common
uniform int receiveShadow;
uniform mat4 modelMatrix;
uniform mat4 modelMatrix_InverseTranspose;

//output
out float shadow;

//should be moved to FS
float GetPointLightShadow(int lightIndex, vec3 pointWorldPos, vec3 worldNormal){
	
	float shadow = 0.0f;
	vec3 lightToPos = PointLightPositions[lightIndex] - pointWorldPos; //到點光源的向量
	float currentDepth = length(lightToPos); //到光源的半徑距離，作為深度值
	vec3 lightDir = normalize(lightToPos); //光源方向
	float bias = max(0.025 * (1.0- dot(worldNormal, lightDir)), 0.005); //根據梯度修正的bias

	//比例近似濾波器
	int sampleDistance = 3;
	vec3 texelSize = 1.0 / vec3(textureSize(shadowMapTextureCubeArray_PointLight, 0));
	for(int z = -sampleDistance; z <= sampleDistance; z++){
		for(int y = -sampleDistance; y <= sampleDistance; y++){
		    for(int x = -sampleDistance; x <= sampleDistance; x++){
		        float closestDepth = texture(shadowMapTextureCubeArray_PointLight, vec4(lightToPos + vec3(x, y, z) * texelSize, lightIndex)).r;
				closestDepth *= PointLightFarPlanes[lightIndex]; //因為創建的時候除了far plane, 所以需要乘回去
				if (currentDepth > closestDepth + bias) shadow += 1.0f;     
		    }    
		}
	}
	// Average shadow
	shadow /= pow((sampleDistance * 2 + 1), 3);
	return shadow;
}	
	
float GetDirectionalLightShadow(int lightIndex, vec3 pointWorldPos, vec3 worldNormal)
{
	float shadow = 0.0f;
	vec3 shadowMapCoords = (DirectLightVP_Bias[lightIndex] * vec4(pointWorldPos,1.0)).xyz;
	vec3 lightDirection = DirectionalLightDirections[lightIndex];
	if(shadowMapCoords.z <= 1.0f)
	{
		float currentDepth = shadowMapCoords.z;
		float bias = max(0.025f * (1.0f - dot(worldNormal, lightDirection)), 0.005);
		
		int sampleDistance = 2;
		vec2 pixelSize = 1.0 / vec2(textureSize(shadowMapTexture2DArray_DirectionalLight, 0));
		for(int y = -sampleDistance; y <= sampleDistance; y++){
		    for(int x = -sampleDistance; x <= sampleDistance; x++){
		        float closestDepth = texture(shadowMapTexture2DArray_DirectionalLight, vec3(shadowMapCoords.xy + vec2(x, y) * pixelSize, lightIndex)).r;
				if (currentDepth > closestDepth + bias)
					shadow += 1.0f;     
		    }    
		}
		shadow /= pow((sampleDistance * 2 + 1), 2);
	}
	return shadow;
}

float GetSpotLightShadow(int lightIndex, vec3 pointWorldPos, vec3 worldNormal)
{
	float shadow = 0.0f;
	vec3 shadowMapCoords = (SpotLightVP_Bias[lightIndex] * vec4(pointWorldPos,1.0)).xyz;
	vec3 lightDirection = SpotLightPositions[lightIndex] - pointWorldPos;
	if(shadowMapCoords.z <= 1.0f)
	{
		float currentDepth = shadowMapCoords.z;
		float bias = max(0.025f * (1.0f - dot(worldNormal, lightDirection)), 0.005);

		int sampleDistance = 2;
		vec2 pixelSize = 1.0 / vec2(textureSize(shadowMapTexture2DArray_SpotLight, 0));
		for(int y = -sampleDistance; y <= sampleDistance; y++){
		    for(int x = -sampleDistance; x <= sampleDistance; x++){
		        float closestDepth = texture(shadowMapTexture2DArray_SpotLight, vec3(shadowMapCoords.xy + vec2(x, y) * pixelSize, lightIndex)).r;
				if (currentDepth > closestDepth + bias)
					shadow += 1.0f;
		    }    
		}
		shadow /= pow((sampleDistance * 2 + 1), 2);
	}
	return shadow;
}

void main()
{
	vec3 worldPos = (modelMatrix * vec4(vertexPos, 1.0)).xyz;
	vec3 realNormal = texture(normal, vertexUV).rgb * 2.0f - 1.0f;
	vec3 worldNormal = vec3(modelMatrix_InverseTranspose * vec4(realNormal,1.0));
	gl_Position = camProjMatrix * camViewMatrix * vec4(worldPos, 1.0);
	shadow = 0.0;
	if (receiveShadow ==1){
		for (int i = 0; i < DirectionalLightShadowCount; i++){
			float tempShadow = GetDirectionalLightShadow(i, worldPos, worldNormal);
			if (tempShadow > shadow)
				shadow = tempShadow;
		}
		for (int i = 0; i < SpotLightShadowCount; i++){
			float tempShadow = GetSpotLightShadow(i, worldPos, worldNormal);
			if (tempShadow > shadow)
				shadow = tempShadow;
		}
		for (int i = 0; i < PointLightShadowCount; i++){
			float tempShadow = GetPointLightShadow(i, worldPos, worldNormal);
			if (tempShadow > shadow)
				shadow = tempShadow;
		}
	}
}

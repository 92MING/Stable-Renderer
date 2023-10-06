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

struct SingleLightData{
	int lightType;
	vec3 direction ; //for directional & spot light
	vec3 position ; //for point & spot light
	float cutOff ; //for spot light
	float outerCutOff; //for spot light
	float constant;
	float linear;
	float quadratic;
	vec3 lightColor;
	float intensity;
};
//lightData
layout (std430, binding = 1) buffer LightData {
	float ambientPower;
    vec3 ambientLight;
    int singleLightDataLength;
	SingleLightData allSingleLightData[MAX_LIGHT_NUM];  
} ;

//data of mesh
in layout(location=0) vec3 vertexPos;
in layout(location=1) vec2 vertexUV;
in layout(location=2) vec3 vertexNormal;

//textures
uniform sampler2D diffuse;
uniform sampler2D normal;
uniform sampler2D metallic;
uniform sampler2D roughness;
uniform sampler2D ao;

//common
uniform int receiveShadow;
uniform mat4 modelMatrix;
uniform mat4 modelMatrix_InverseTranspose;
uniform vec3 emissionColor;
uniform float emissionIntensity;

in float shadow; //from VS
out vec4 fragColor;

vec3 GetDirectionalLightColor(SingleLightData thisLight, vec3 worldPos, vec3 worldNormal, vec3 diffuseColor, float roughFactor, float metalFactor)
{
	float diffuseFactor =  max(dot(worldNormal, thisLight.direction), 0.0f);
	vec3 diffuseTerm = diffuseFactor * diffuseColor;
	vec3 specularTerm = vec3(0.0,0.0,0.0);
	if (diffuseFactor != 0.0f)
	{
		float specularLight = 0.50f;
		vec3 viewDirection = normalize(camPos - worldPos);
		vec3 halfDir = normalize( (-thisLight.direction) + viewDirection);
		specularTerm =  metalFactor * diffuseColor * pow(max(dot(worldNormal, halfDir), 0.0), roughFactor);
	};
	return diffuseTerm + specularTerm; 
}

vec3 GetPointLightColor(SingleLightData thisLight, vec3 worldPos, vec3 worldNormal, vec3 diffuseColor, float roughFactor, float metalFactor){
	vec3 pointToLightDir = normalize(thisLight.position - worldPos);
	float diffuseFactor =  max(dot(worldNormal, pointToLightDir), 0.0f);
	vec3 diffuseTerm = diffuseFactor * diffuseColor;
	vec3 specularTerm = vec3(0.0,0.0,0.0);
	if (diffuseFactor != 0.0f)
	{
		float specularLight = 0.50f;
		vec3 viewDirection = normalize(camPos - worldPos);
		vec3 halfDir = normalize( pointToLightDir + viewDirection);
		specularTerm =  metalFactor * diffuseColor * pow(max(dot(worldNormal, halfDir), 0.0), roughFactor);
	};
	float lightDistance = length(thisLight.position - worldPos);
	float displaceFactor = thisLight.quadratic * pow(lightDistance, 2) + thisLight.linear * lightDistance + thisLight.constant;
	return (1/displaceFactor) * (diffuseTerm + specularTerm); 
}

vec3 GetSpotLightColor(SingleLightData thisLight, vec3 worldPos, vec3 worldNormal, vec3 diffuseColor, float roughFactor, float metalFactor)
{
	vec3 pointToLightDir = normalize(thisLight.position - worldPos);
	float angle = acos(dot(pointToLightDir, worldNormal)) / 3.14159;
	if (angle <0 || angle>thisLight.outerCutOff) return vec3(0.0,0.0,0.0);
	
	float diffuseFactor =  max(dot(worldNormal, pointToLightDir), 0.0f);
	vec3 diffuseTerm = diffuseFactor * diffuseColor;
	vec3 specularTerm = vec3(0.0,0.0,0.0);
	if (diffuseFactor != 0.0f)
	{
		float specularLight = 0.50f;
		vec3 viewDirection = normalize(camPos - worldPos);
		vec3 halfDir = normalize( pointToLightDir + viewDirection);
		specularTerm =  metalFactor * diffuseColor * pow(max(dot(worldNormal, halfDir), 0.0), roughFactor);
	};
	float lightDistance = length(thisLight.position - worldPos);
	float displaceFactor = thisLight.quadratic * pow(lightDistance, 2) + thisLight.linear * lightDistance + thisLight.constant;
	float intensity = clamp((angle - thisLight.outerCutOff) / (thisLight.cutOff - thisLight.outerCutOff), 0.0f, 1.0f);
	return (1/displaceFactor) * intensity * (diffuseTerm + specularTerm); 
}


void main()
{
	vec4 worldPos = modelMatrix * vec4(vertexPos,1.0);
	vec4 worldNormal = modelMatrix_InverseTranspose * vec4(vertexNormal,1.0);
	vec4 diffuseColor = texture(diffuse, vertexUV);
	float roughFactor = texture(roughness, vertexUV).r;
	float metalFactor = texture(metallic, vertexUV).r;
	float alpha = diffuseColor.a;
	
	float aoFactor = texture(ao, vertexUV).r;
	vec3 allLightColor = vec3(0.0,0.0,0.0);
	for(int i=0; i< singleLightDataLength;i++){
		if (allSingleLightData[i].lightType == POINT_LIGHT) {
			allLightColor += GetPointLightColor(allSingleLightData[i], worldPos.xyz, worldNormal.xyz, diffuseColor.xyz, roughFactor, metalFactor);
		}
		else if (allSingleLightData[i].lightType == SPOT_LIGHT) {
			allLightColor += GetSpotLightColor(allSingleLightData[i], worldPos.xyz, worldNormal.xyz, diffuseColor.xyz, roughFactor, metalFactor);
		}
		else if (allSingleLightData[i].lightType == DIRECTIONAL_LIGHT) {
			allLightColor += GetDirectionalLightColor(allSingleLightData[i], worldPos.xyz, worldNormal.xyz, diffuseColor.xyz, roughFactor, metalFactor);
		}
	}
	//fragColor = vec4(aoFactor * ambientLight * ambientPower + emissionIntensity * emissionColor + (1-shadow) * allLightColor ,alpha);
	fragColor = vec4(1.0,1.0,1.0,1.0);
}
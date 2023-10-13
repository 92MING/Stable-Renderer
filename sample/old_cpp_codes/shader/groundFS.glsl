#version 430
uniform mat4 modelMatrix;
uniform vec3 camPos;
uniform vec3 ambient;

//directional light
uniform sampler2D direct_shadowMap;
uniform vec3 direct_lightCol;
uniform vec3 direct_lightDir;
uniform float direct_lightPow;
uniform mat4 direct_lightMatrix_bias;

//spot light
uniform sampler2D spot_shadowMap;
uniform vec3 spot_lightCol;
uniform vec3 spot_lightDir;
uniform float spot_lightPow;
uniform float spot_lightOut;
uniform float spot_lightIn;
uniform vec3 spot_lightPos;
uniform mat4 spot_lightMatrix_bias;

//point light
uniform samplerCube point_shadowMap;
uniform vec3 point_lightCol;
uniform float point_lightPow;
uniform vec3 point_lightPos;
uniform float pointLightFarPlane;

vec2 poissonDisk[4] = vec2[](
    vec2(-0.94201624, -0.39906216),
    vec2(0.94558609, -0.76890725),
    vec2(-0.094184101, -0.92938870),
    vec2(0.34495938, 0.29387760)
);
int poissonSample = 4;

in vec3 worldPos;
in vec3 worldNormal;
in vec3 VScolor;
//in float depth;
out vec4 theColor;

void main()
{
    float textureLength = textureSize(direct_shadowMap, 0).x;
    float texelLength = 1.0 / float(textureLength);
    vec3 viewDir = normalize(camPos - worldPos);
    vec3 baseColor = vec3(1.0,1.0,1.0);

    float ao=1.0;
    float roughness = 4.0;
    float metallic = 1.5;
    
    //directional light
    float diffuseFactor = max(dot(worldNormal, -direct_lightDir), 0.0);
    vec4 lightSpacePos = direct_lightMatrix_bias * vec4(worldPos,1.0);
    float bias = max(0.0125 * (1.0 - dot(worldNormal, -direct_lightDir)), 0.005);
    float direct_shadow = 0.0;
    if (lightSpacePos.z<1.0){
        int sampleRadius = 2;
        float total = 0.0;
        for(int x = -sampleRadius; x <= sampleRadius; ++x){
		    for(int y = -sampleRadius; y <= sampleRadius; ++y){
                for (int i=0;i<poissonSample;i++){
                    if (texture(direct_shadowMap ,lightSpacePos.xy + vec2(x, y)* texelLength + poissonDisk[i]/700).r < lightSpacePos.z - bias)
				        total += 1.0/poissonSample;
                }
		    }
	    }
        direct_shadow = total / ((sampleRadius * 2 + 1) * (sampleRadius * 2 + 1));
        direct_shadow *= direct_shadow*direct_lightPow;
     }
    vec3 halfDir = normalize(viewDir + direct_lightDir);
    float spec = metallic * pow(max(dot(worldNormal, halfDir), 0.0), roughness);
    vec3 direct_light = direct_lightCol * direct_lightPow * (1-direct_shadow)  * (diffuseFactor + spec) * baseColor;

	//spot light
    vec3 spot_light = vec3(0.0);
    diffuseFactor = 0.0;
    float angle = acos(dot(normalize(worldPos.xyz - spot_lightPos), spot_lightDir)) * 180.0 / 3.1415926;
    if (angle < spot_lightOut){
        vec3 pointToLight = normalize(spot_lightPos - worldPos.xyz);
        diffuseFactor = max(dot(worldNormal, pointToLight), 0.0);
        lightSpacePos = spot_lightMatrix_bias * vec4(worldPos,1.0);
	    bias = max(0.05 * (1.0 - dot(worldNormal, pointToLight)), 0.005);
	    float spot_shadow = 0.0;
        if ((lightSpacePos.z/lightSpacePos.w)<1.0)
        {
	        int sampleRadius = 2;
	        float total = 0.0;
	        for(int x = -sampleRadius; x <= sampleRadius; ++x){
		        for(int y = -sampleRadius; y <= sampleRadius; ++y){
			        for (int i=0;i<poissonSample;i++){
                        if (texture(spot_shadowMap ,(lightSpacePos.xy+ vec2(x, y)* texelLength)/lightSpacePos.w + poissonDisk[i]/700).r < (lightSpacePos.z - bias)/lightSpacePos.w)
				            total += 1.0/poissonSample;
                    }
		        }
	        }
            spot_shadow = total / ((sampleRadius * 2 + 1) * (sampleRadius * 2 + 1));
            spot_shadow *=  (spot_lightPow / length(worldPos - spot_lightPos));
        }
	    halfDir = normalize(viewDir + spot_lightDir);
	    spec = metallic * pow(max(dot(worldNormal, halfDir), 0.0), roughness);
        float distanceFactor = 1.0f/ pow(length(worldPos - spot_lightPos),1.5);
        if (angle> spot_lightIn){
			distanceFactor = distanceFactor * (angle - spot_lightOut)/(spot_lightIn - spot_lightOut);
        }
        spot_light = spot_lightCol * spot_lightPow * distanceFactor * (1-spot_shadow)  * (diffuseFactor + spec) * baseColor;
    }
    
    //point light
    vec3 point_light = vec3(0.0);
    diffuseFactor = 0.0;
    vec3 fragToLight = (worldPos - point_lightPos);
    float currentDepth = length(fragToLight);
    bias = max(0.05 * (1.0 - dot(worldNormal, -normalize(fragToLight))), 0.005) ;
    float point_shadow = 0.0;
    if (currentDepth < pointLightFarPlane){
        diffuseFactor = max(dot(worldNormal, -normalize(fragToLight)), 0.0);
        int sampleRadius = 3;
        float total = 0.0;
        for(int x = -sampleRadius; x <= sampleRadius; ++x){
            for(int y = -sampleRadius; y <= sampleRadius; ++y){
                for(int z = -sampleRadius; z <= sampleRadius; ++z){
                    for (int i=0;i<poissonSample;i++){
                        vec3 samplePos = fragToLight + vec3(x, y, z)*texelLength + vec3(poissonDisk[i]/700, 0.0);
			            if (texture(point_shadowMap,samplePos).r * pointLightFarPlane < currentDepth - bias)
				            total += 1.0/poissonSample;
                    }
		        }
	        }
	    }
        point_shadow = total / pow((sampleRadius * 2 + 1),3);
        point_shadow *= (point_lightPow / length(worldPos - point_lightPos));
    }
    halfDir = normalize(viewDir + normalize(fragToLight));
	spec = metallic * pow(max(dot(worldNormal, halfDir), 0.0), roughness);
	float distanceFactor = 1.0f/ pow(length(worldPos - point_lightPos),1.5);
	point_light = point_lightCol * point_lightPow * distanceFactor * (1- point_shadow )* ( diffuseFactor + spec) * baseColor;
    
	//color
	theColor =  vec4((direct_light + spot_light + point_light +ambient ),0.5) ;
    //theColor = vec4(depth);
}

